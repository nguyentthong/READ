import torch
import torch.nn as nn
from nncore.nn import MODELS, build_model, xavier_init_
import nncore
from geomloss.sinkhorn_samples import cost_routines

def partial_ot(C, a, b, beta=1000, n_iter=10):
    m = a.shape[0]
    n = b.shape[0]
    a = a.cuda()
    b = b.cuda()
    C = C.cuda()
    one_m = torch.ones(m, 1).cuda()
    one_n = torch.ones(n, 1).cuda()
    N_V = a.shape[0]
    N_L = b.shape[0]

    for s in range(1, min(N_V, N_L)+1):
        D = torch.tensor(torch.inf).cuda()
        
        T = torch.exp(-C/beta).cuda()
        T = (s/(torch.matmul(torch.matmul(one_m.T, T), one_n))).item() * T
        
        for t in range(n_iter):
            k_a = torch.min(torch.div(a, torch.matmul(T, one_n)), one_m)
            T_a = torch.diag(k_a) * T
            k_b = torch.min(torch.div(b, torch.matmul(T_a.T, one_m)), one_n)
            T_b = torch.diag(k_b) * T_a
            T = (s/(torch.matmul(torch.matmul(one_m.T, T), one_n))).item() * T_b

        D = torch.min(D, torch.dot(C.flatten(), T.flatten()))
    return D

@MODELS.register()
class UMT(nn.Module):

    def __init__(self,
                 video_enc=None,
                 audio_enc=None,
                 cross_enc=None,
                 query_gen=None,
                 query_dec=None,
                 pred_head=None,
                 p=2,
                 gamma=1e-2):
        super(UMT, self).__init__()

        cnt = sum(e is None for e in (video_enc, audio_enc, cross_enc))
        assert not cnt % 2 and ((query_gen is None) == (query_dec is None))

        self.video_enc = build_model(video_enc)
        self.audio_enc = build_model(audio_enc)
        self.cross_enc = build_model(cross_enc)
        self.query_gen = build_model(query_gen)
        self.query_dec = build_model(query_dec)
        self.pred_head = build_model(pred_head, bundler='modulelist')

        self.apply(lambda m: xavier_init_(m)
                   if isinstance(m, nn.Linear) else None)

        self.cost = cost_routines[p]
        self.gamma = gamma

    def evaluate(self, data, blob, **kwargs):
        collected = []

        inds = torch.argsort(blob['saliency'], descending=True)
        label = data['saliency'][0][inds].tolist()[0]

        if (num_gt := sum(label)) == 0:
            collected.append(0)
            return 0.0

        hits = ap = rec = 0
        prc = 1

        for i, gt in enumerate(label):
            hits += gt

            _rec = hits / num_gt
            _prc = hits / (i + 1)

            ap += (_rec - rec) * (prc + _prc) / 2
            rec, prc = _rec, _prc

        collected.append(ap)

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))

        return results

    def calc_pot_loss(self, v_emb, a_emb):
        m = v_emb.shape[1]
        n = a_emb.shape[1]

        C_va = self.cost(v_emb, a_emb)
        distance_list = []
        for i in range(len(C_va)):
            current_C_va = C_va[i]
            a = torch.ones(m, 1)/m
            b = torch.ones(n, 1)/n
            distance = partial_ot(current_C_va, a, b)
            distance_list.append(distance)
        distance = torch.mean(torch.stack(distance_list))
        return distance

    def forward(self, data, mode):
        if isinstance(data['saliency'], nncore.parallel.container.DataContainer):
            for key in data:
                data[key] = data[key].data[0]

        mask = torch.where(data['saliency'] >= 0, 1, 0).cuda()

        if self.video_enc is not None:
            d_emb = r_emb = v_emb = self.video_enc(data['video'].cuda(), mask=mask)
        else:
            v_emb = data['video']

        if self.audio_enc is not None:
            d_emb = r_emb = a_emb = self.audio_enc(data['audio'].cuda(), mask=mask)
        else:
            a_emb = data['audio']

        if self.cross_enc is not None:
            d_emb = r_emb = self.cross_enc(v_emb, a_emb, mask=mask)

            if self.training:
                pot_loss = self.calc_pot_loss(v_emb, a_emb)
        
        if self.query_gen is not None:
            try:
                q_emb = self.query_gen(r_emb, data.get('query').cuda())
            except:
                q_emb = self.query_gen(r_emb, data.get('query'))

            d_emb = self.query_dec(q_emb, r_emb)

        output = dict(_avg_factor=mask.size(0), _out=dict(meta=data.get('meta')))

        for pred_head in self.pred_head:
            output = pred_head(d_emb, data, output, mode)
        
        results = self.evaluate(data, output['_out'])
        if self.training:
            for key in output:
                if 'loss' in key:
                    output[key] = output[key] + pot_loss * self.gamma

        return output
