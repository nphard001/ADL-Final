from host_data.views import *

# import torch
# from fashion_retrieval_old import sim_user, ranker
# from fashion_retrieval_old.model import NetSynUser
def _ApplyURLPatterns():
    urlpatterns.append(path(r'predict', predict_view))
# ================================================================
# class ModelTester(dict):
#     def __init__(self):
#         # Load user and ranker
#         user = sim_user.SynUser()
#         my_ranker = ranker.Ranker()

#         # Create inverse mapping for self-defined sentences
#         inv_map = {v: k for k, v in user.captioner_relative.vocab.items()}
        
#         model_path = r'fashion_retrieval_old/models/rl-13.pt'
#         model = NetSynUser(user.vocabSize + 1)
#         # model.load_state_dict(torch.load(model_path))
        
#         self.user = user
#         self.my_ranker = my_ranker
#         self.inv_map = inv_map
#         self.model_path = model_path
#         self.model = model
        
@csrf_exempt
def predict_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    return JsonResponse({
        'state': 'WIP',
        'original_json': json_body,
    })

# ================================================================
_ApplyURLPatterns()