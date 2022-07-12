from sentence_transformers import util

def accumulate_predictions(dev_examples, pred):
    pred_sum_arr_p1 = []
    pred_sum_arr_p2 = []
    for d in dev_examples:
        pred_sum_arr_p1.append(pred[d.ID]['Person1'])
        pred_sum_arr_p2.append(pred[d.ID]['Person2'])
        
    return pred_sum_arr_p1, pred_sum_arr_p2
        
def similar(p1_sents, p2_sents, model):
    embeddings_p1 = model.encode(p1_sents, convert_to_tensor=True)
    embeddings_p2 = model.encode(p2_sents, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings_p1, embeddings_p2)
    
    return cosine_scores.cpu()