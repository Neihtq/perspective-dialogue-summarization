import json
from tqdm import tqdm
from src.data.dataset import InputExample, InputFeatures
from src.utils.constants import DIALOGSUM_SEGMENTATION_PATH, BOS_TOKEN, HL_TOKEN, SUM_TOKEN, MINI_DIALOGSUM_SEGMENTATION_PATH


def removeElements(A, B): 
    n = len(A) 
    return any(A == B[i:i + n] for i in range(len(B)-n + 1))


def locate_sublist(sublist, parent):
    cursor = 0
    for i, ele in enumerate(parent):
        if ele == sublist[0]:
            if parent[i: i + len(sublist)] == sublist:
                cursor = i
                break
    return cursor, cursor + len(sublist)


def get_intent(index):
    # WHY = 0; WHAT = 1; WHERE = 2; WHEN = 3; CONFIRM = 4; ABSTAIN = -1
    intent_dict = {0:"why", 1:"what", 2:"where", 3:"when", 4:"confirm", -1:"abstain"}
    return intent_dict[index]


def get_func_turn_label(args, config, source_ids, example, person, max_num_of_turns=50):
    '''Get functional turns label (truncate to max_len), either only 0/1 or 0-6 modular index'''
    example_func_turn_label = example.func_turn_label_p1 if person == 'p1' else example.func_turn_label_p2
    local_max_num_of_turns = source_ids.count(config.bos_token_id)
    if args.add_module_loss:
        func_turn_label = []
        counter = 0
        for ftl_i, ftl in enumerate(func_turn_label):
            if (ftl == 1) or \
                (ftl_i > 0 and example_func_turn_label[ftl_i-1] == 1) or \
                (ftl_i < len(example_func_turn_label)-1 and example_func_turn_label[ftl_i+1] == 1):
                func_turn_label.append(example.module_index[counter]+2)
                counter += 1
            else:
                func_turn_label.append(0)                
        assert len([i for i in func_turn_label if i!=0]) == len(example.module_index)
    elif args.add_functurn_loss:
        func_turn_label = example_func_turn_label
    else:
        func_turn_label = [-1] * max_num_of_turns
    func_turn_label = func_turn_label[:local_max_num_of_turns][:max_num_of_turns]
    padding_len = max_num_of_turns - local_max_num_of_turns
    func_turn_label += ([-1] * padding_len)

    return func_turn_label


def convert_examples_to_features(args, config, tokenizer, examples):
    features = []
    max_target_len = index = 0

    for e in tqdm(examples, desc='Examples'):
        # Process source information
        source = e.context
        source_tokens = tokenizer.tokenize(source)[:args.source_max_len-2]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) + [config.eos_token_id] # <s> ... </s>
        source_len = len(source_ids)
        source_mask = [1] * source_len
        padding_len = args.source_max_len - source_len
        source_ids += ([config.pad_token_id] * padding_len)
        source_mask += ([0] * padding_len)
        assert len(source_ids) == args.source_max_len
        assert len(source_mask) == args.source_max_len
        
        # Process target information
        answer_tokens_p1 = get_answer_tokens(args, tokenizer, e, 'p1', source_ids, config, max_target_len)
        answer_tokens_p2 = get_answer_tokens(args, tokenizer, e, 'p2', source_ids, config, max_target_len)

        target_ids_p1, target_labels_p1, target_len_p1 = process_target_information(args, config, tokenizer, answer_tokens_p1)
        target_ids_p2, target_labels_p2, target_len_p2 = process_target_information(args, config, tokenizer, answer_tokens_p2)

        func_turn_label_p1 = get_func_turn_label(args, config, source_ids, e, 'p1')
        func_turn_label_p2 = get_func_turn_label(args, config, source_ids, e, 'p2')
        f = InputFeatures(e.ID, 
            index, source_ids,
            source_mask, source_len, 
            target_ids_p1, target_labels_p1, target_len_p1,
            target_ids_p2, target_labels_p2, target_len_p2,  
            func_turn_label_p1, func_turn_label_p2)
        features.append(f)

        index += 1
    
    print("[INFO] max_target_len", max_target_len)
    return features


def process_target_information(args, config, tokenizer, answer_tokens):
    answer_tokens_ = tokenizer.convert_tokens_to_ids(answer_tokens)
    target_ids = [config.bos_token_id] + answer_tokens_ # <s> ...
    target_labels = answer_tokens_ + [config.eos_token_id] # ... </s>
    target_len = len(target_ids)
    padding_len = args.target_max_len - target_len
    target_ids += ([config.pad_token_id] * padding_len)
    target_labels += ([-100] * padding_len) # -100 is the default index to be ignored
    assert len(target_ids) == args.target_max_len
    assert len(target_labels) == args.target_max_len

    return target_ids, target_labels, target_len


def get_answer_tokens(args, tokenizer, example, person, source_ids, config, max_target_len, max_nb_turns_kp=20):
    summary = example.summary_p1 if person == 'p1' else example.summary_p2
    if args.gen_keyphrase_summary:
        string_global = []
        key_phrases = example.key_phrases_p1[:max_nb_turns_kp] if person == 'p1' else example.key_phrases_p2[:max_nb_turns_kp] 
        for ki, key_phrases in enumerate(key_phrases):
            if len(key_phrases) > 0:
                string = [str(ki), get_intent(example.module_index[ki])] + key_phrases
                string = " ".join(string)
                string_global.append(string)
            else:
                string_global.append("{} {}".format(ki, "none"))

        string_global = " ".join(string_global[:source_ids.count(config.bos_token_id)])
        string_output = string_global + " {} ".format(SUM_TOKEN) + summary
        answer_tokens = tokenizer.tokenize(string_output)
        if len(answer_tokens) > max_target_len: max_target_len = len(answer_tokens)
        answer_tokens = answer_tokens[-args.target_max_len+1:] # -1 for <s> or </s>
    else:
        string_output = summary
        answer_tokens = tokenizer.tokenize(string_output)
        if len(answer_tokens) > max_target_len: max_target_len = len(answer_tokens)
        answer_tokens = answer_tokens[:args.target_max_len-1] # -1 for <s> or </s>
    
    # -1 for <s> or </s>
    if args.gen_keyphrase_summary:
        return answer_tokens[-args.target_max_len+1:]

    return answer_tokens[:args.target_max_len-1], max_target_len


def process_segmentation_results(data, examples, context, segment_labels, summaries, example_args):
    # Process input and output for different segmentation results
    seg_count_1, seg_count_2, seg_idx_1, seg_idx_2 = 0, 0, 0, 0
    _bos_token_ = " {} ".format(BOS_TOKEN)
    if sum(segment_labels[0]) == 0 and sum(segment_labels[1]) == 0:
        context = "{} {} {} {}".format(BOS_TOKEN, HL_TOKEN, context, HL_TOKEN)
        e = InputExample(ID="{}#{}".format(data["id"], (seg_count_1 + seg_count_2)),
                context=context,
                summary_p1=summaries[0], summary_p2=summaries[1],
                func_turn_label_p1=example_args[2], func_turn_label_p2=example_args[3],
                key_phrases_p1=example_args[4], key_phrases_p2=example_args[5],
                module_index=example_args[6]
            )
        examples.append(e)
    else:
        for si, (seg_l_1, seg_l_2) in enumerate(zip(segment_labels[0], segment_labels[1])):
            if seg_l_1 == 1 or seg_l_2 == 1 or si == len(segment_labels[0]) - 1:
                seg_idx = seg_idx_1 if seg_l_1 == 1 else seg_idx_2
                if si == len(segment_labels[0]) - 1:
                    seg_idx = -1
                temp = list(data["clean_dialog"])
                temp[seg_idx] = "{} {}".format(HL_TOKEN, temp[seg_idx])
                temp[si] = "{} {}".format(temp[si], HL_TOKEN)
                context = "{} ".format(BOS_TOKEN) + _bos_token_.join(temp)

                e = InputExample(ID="{}#{}".format(data["id"], (seg_count_1 + seg_count_2)),
                        context=context,
                        summary_p1=example_args[0][seg_count_1], summary_p2=example_args[1][seg_count_2],
                        func_turn_label_p1=example_args[2], func_turn_label_p2=example_args[3],
                        key_phrases_p1=example_args[4][seg_idx_1:si], key_phrases_p2=example_args[5][seg_idx_2:si],
                        module_index=example_args[6],
                    )
                examples.append(e)
                if seg_l_1 == 1:
                    seg_idx_1 = si + 1
                    seg_count_1 += 1
                if seg_l_2 == 1:
                    seg_idx_2 = si + 1
                    seg_count_2 += 1


def get_segment_label(args, pred_segment_dict, data, person):
    if args.use_pred_segment:
        segment_label = pred_segment_dict[data["id"]][f"segment_label_{person}"]
        if args.ctrl_nb_summary_sent:
            if args.ctrl_nb_summary_sent == 1:
                segment_label = [0 for _ in segment_label]
            elif args.ctrl_nb_summary_sent >= len(segment_label):
                pass
            else:
                segment_prob = np.array(pred_segment_dict[data["id"]][f"segment_prob_{person}"])
                topk_idx = segment_prob.argsort()[-args.ctrl_nb_summary_sent+1:][::-1]
                segment_label = [1 if i in topk_idx else 0 for i in range(len(segment_label))]
        sum_list = ["summary"] * (sum(segment_label) + 1)
    else:
        segment_label = data[f"segment_label_{person}"]
        sum_list = data[f"sum_list_{person}"]
    
    return segment_label, sum_list


def get_labels(data, person):
    return data[f'label_p{person}'], data[f'module_index_{person}'], data[f'key_phrases_p{person}']


def cleanup_summary(data, person):
    to_continue = False
    summary = data[f"summary_{person}"].replace("\n", " ").replace("\015", "")
    if len(summary.strip()) < 5: # there are several error summary in train set
        print("[WARNING] Skip summary [{}]".format(summary))
        to_continue = True
    
    return summary, to_continue


def load_examples(args, file_path):
    examples = []
    # Get predicted segmentation for inference
    pred_segment_dict = {}
    if args.use_pred_segment:
        assert args.do_train == False
        with open(MINI_DIALOGSUM_SEGMENTATION_PATH, 'r') as f:
        #with open(DIALOGSUM_SEGMENTATION_PATH, 'r') as f:
            pred_segment_dict = json.load(f)
    
    # Data reading
    with open(file_path, 'r') as f:
        jsn = json.load(f)
        for data in jsn:
            summary_p1, to_continue_1 = cleanup_summary(data, 'p1')
            summary_p2, to_continue_2 = cleanup_summary(data, 'p2')
            if to_continue_1 or to_continue_2: continue

            _bos_token_ = " {} ".format(BOS_TOKEN)
            if args.oracle_functurn_context:
                context = "{} ".format(BOS_TOKEN) + _bos_token_.join(data["function_dialogs"]).replace("\n", " ").replace("\015", "")
            else:
                context = "{} ".format(BOS_TOKEN) + _bos_token_.join(data["clean_dialog"]).replace("\n", " ").replace("\015", "")

            func_turn_label_p1, module_index_p1, key_phrases_p1 = get_labels(data, '1')
            func_turn_label_p2, module_index_p2, key_phrases_p2 = get_labels(data, '2')

            if args.do_segment:
                # Whether use predicted segmentation or control nb of summary sentences
                segment_label_p1, sum_list_p1 = get_segment_label(args, pred_segment_dict, data, '1')
                segment_label_p2, sum_list_p2 = get_segment_label(args, pred_segment_dict, data, '2')
                
                process_segmentation_results(data, examples, context,
                    [segment_label_p1, segment_label_p2],
                    [summary_p1, summary_p2],
                    [sum_list_p1, sum_list_p2, func_turn_label_p1, func_turn_label_p2, key_phrases_p1, key_phrases_p1, module_index_p1] 
                ) 
            else: 
                e = InputExample(ID=data["id"],
                        context=context,
                        summary_p1=summary_p1, summary_p2=summary_p2,
                        func_turn_label_p1=func_turn_label_p1, func_turn_label_p2=func_turn_label_p1,
                        key_phrases_p1=key_phrases_p1, key_phrases_p2=key_phrases_p2,
                        module_index=module_index_p1
                    )
                examples.append(e)
    '''
    print('\n', file_path, len(examples))
    print("examples[0].ID", examples[0].ID)
    print("examples[0].context", examples[0].context)
    print("examples[0].summary_p1", examples[0].summary_p1)
    print("examples[0].func_turn_label_p1", examples[0].func_turn_label_p1)
    print("examples[0].key_phrases_p1", examples[0].key_phrases_p1, '\n')
    print("examples[0].summary_p2", examples[0].summary_p2)
    print("examples[0].func_turn_label_p2", examples[0].func_turn_label_p2)
    print("examples[0].key_phrases_p2", examples[0].key_phrases_p2, '\n')
    print("examples[0].module_index", examples[0].module_index)
    '''
    return examples
