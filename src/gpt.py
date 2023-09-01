import os
import openai
import json
from tqdm import tqdm
import time
import sys
from wiki_api import WikiSearchAPI

openai.api_key = os.getenv("OPENAI_API_KEY")

# data preparation
# claim_list = []
# s_cnt = 0
# ns_cnt = 0
# cnt = 0
# selected_dev_data = []
# with open("dev.json", "r") as reader:
#     dev_data = json.load(reader)
#     writer = open("dev_selected.json", "w")
#     for i in range(len(dev_data)):
#         if (i % 4) == 0:
#             claim_list.append(dev_data[i]["claim"])
#             cnt += 1 
#             js = {}
#             js["id"] = str(cnt)
#             js["claim"] = dev_data[i]["claim"]
#             js["label"] = dev_data[i]["label"]
#             writer.write(json.dumps(js))
#             writer.write("\n")
#             js["facts"] = dev_data[i]["supporting_facts_contain_text"]
#             selected_dev_data.append(js)
#             if(dev_data[i]["label"] == "SUPPORTED"):
#                 s_cnt += 1
#             else:
#                 ns_cnt += 1
#     writer.close() # 一定要记得close()，否则写入的内容可能一部分还在缓冲区里，没有写入文件！！


'''
data preparation
'''
def prepara_data(dev_file_path):
    selected_dev_data = []
    cnt = 0
    with open(dev_file_path, "r") as reader:
        dev_data = json.load(reader)
        for i in range(len(dev_data)):
            if (i % 4) == 0:
                cnt += 1
                js = {}
                js["id"] = str(cnt)
                js["claim"] = dev_data[i]["claim"]
                js["label"] = dev_data[i]["label"]
                js["facts"] = dev_data[i]["supporting_facts_contain_text"]
                selected_dev_data.append(js)
    claim_list = [d["claim"] for d in selected_dev_data]
    return selected_dev_data, claim_list

selected_dev_data, claim_list = prepara_data("dev.json")

'''
methods for testing
'''

def prompt(system_input, user_input, model_name="gpt-3.5-turbo", show_message=False):
        system_json = {"role": "system", "content": system_input}
        user_json = {"role": "user", "content": user_input}
        message = [system_json, user_json]
        for _ in range(3): # 尝试至多3次
            try:
                # if(show_message == True):
                print(message)
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=message,
                    temperature=0.3,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                time.sleep(1)
        answer = response['choices'][0]['message']['content']
        return answer
        
def multi_round_prompt(system_input, user_input, model_name="gpt-3.5-turbo", show_message=False, few_shot=False):
    # user_input should be a list of strings for multi-round conversation
    history = []
    for round in user_input:
        if len(history) == 0:
            history = [{"role": "system", "content": system_input}]
        history.append({"role": "user", "content": round})
        for _ in range(3): # 尝试至多3次
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=history,
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                time.sleep(1)
        gpt_output = response['choices'][0]['message']['content']
        if few_shot == True:
            # TODO: delete the few-shot examples in the history
            history[-1]['content'] = "Please decompose the claim into several sub-claims.\n claim: " + history[-1]['content'].split("claim: ")[-1]
        history.append({"role": "assistant", "content": gpt_output})
    answers = [item["content"] for item in history if item["role"] == "assistant"]
    if(show_message == True):
        print(history)
    return history, answers
        

def naive_CoT_test(result_path):
    # naive_CoT
    # system_input = "You are a helpful assistant that verifies complex claims"
    # user_input_suffix = "\" First, you need to decompose the claim into several sub-claims. Then, try to verify these sub-claims using all information you know. Finally, give the answer of the verification of the original claim. The answer should be either SUPPORTED or NOT_SUPPORTED."
    
    # naive_CoT update
    system_input = "You are a helpful assistant that verifies complex claims"
    user_input_suffix = "\" First, you need to decompose the claim into several sub-claims. Then, try to verify these sub-claims using all information you know. Finally, give the final answer of verification for the claim. The final answer should be either SUPPORTED or NOT_SUPPORTED. Please give the final answer in the last line."
    
    model_name = "gpt-3.5-turbo"
    with open(result_path, "w") as writer:
        for claim in tqdm(claim_list):
            user_input = "Your task is verify this claim: \"" + claim.strip() + user_input_suffix
            answer = prompt(system_input, user_input, model_name=model_name, show_message=True)
            ans = {}
            ans["answer"] = answer
            writer.write(json.dumps(ans))
            writer.write("\n")

def CoT_noInfoJudge_test(result_path):
    system_input = "You are a helpful assistant that verifies complex claims"
    model_name = "gpt-3.5-turbo"
    current_work = "Your task is verify this claim: Zamalek Sporting Club Centennial was a friendly match against professional football club that plays in La Liga. First, you need to decompose the claim into several sub-claims. Then, try to verify these sub-claims using all information you know. The verification of each sub-claim should be NOT_ENOUGH_INFO if you find no evidence, SUPPORTED if there is evidence supporting it, or REFUTED if there is evidence refuting it. Finally, you need to give the verification of the original claim, which should be NOT_ENOUGH_INFO if any sub-claim lacks evidence, otherwise, SUPPORTED or REFUTED."

def direct_test(result_path):
    system_input = "You are a helpful assistant that verifies complex claims."
    user_input_prefix = "Please verify this claim using all information you know: \""
    model_name = "gpt-3.5-turbo"
    with open(result_path, "w") as writer:
        for claim in tqdm(claim_list):
            user_input = user_input_prefix + claim.strip() + "\" The final answer of verification for the claim should be either SUPPORTED or NOT_SUPPORTED. Please give the final answer in the last line."
            answer = prompt(system_input, user_input, model_name=model_name, show_message=True)
            ans = {}
            ans["answer"] = answer
            writer.write(json.dumps(ans))
            writer.write("\n")

def direct_CoT_test(result_path):
    system_input = "You are a helpful assistant that verifies complex claims."
    user_input_prefix = "Please verify this claim using all information you know: \""
    model_name = "gpt-3.5-turbo"
    with open(result_path, "w") as writer:
        for claim in tqdm(claim_list):
            user_input = user_input_prefix + claim.strip() + "\" The final answer of verification for the claim should be either SUPPORTED or NOT_SUPPORTED. Please give the final answer in the last line. Let's think step by step. "
            answer = prompt(system_input, user_input, model_name=model_name, show_message=True)
            ans = {}
            ans["answer"] = answer
            writer.write(json.dumps(ans))
            writer.write("\n")

def combine_facts(evi_list):
    ret = ""
    for item in evi_list:
        for t in item['text']:
            ret += item['title'] + ": " + t + " "
    return ret

def direct_with_evidence_test(result_path):
    system_input = "You are a helpful assistant that verifies complex claims."
    user_input_prefix = "Please verify this claim: \""
    model_name = "gpt-3.5-turbo"

    with open(result_path, "w") as writer:
        for claim in tqdm(selected_dev_data[:300]):
            facts = combine_facts(claim['facts'])   
            user_input = user_input_prefix + claim['claim'].strip() + "\" The final answer of verification for the claim should be either SUPPORTED or NOT_SUPPORTED. Please give the final answer in the last line. Let's think step by step. Here are some facts for your reference: " + facts
            answer = prompt(system_input, user_input, model_name=model_name, show_message=True)
            ans = {}
            ans["answer"] = answer
            writer.write(json.dumps(ans))
            writer.write("\n")

def CoT_with_evidence_test(result_path):
    system_input = "You are a helpful assistant that verifies complex claims."
    user_input_prefix = "Please verify this claim: \""
    model_name = "gpt-3.5-turbo"
    with open(result_path, "w") as writer:
        for claim in tqdm(selected_dev_data):
            facts = combine_facts(claim['facts'])
            user_input = user_input_prefix + claim['claim'].strip() + "\" First, you need to decompose the claim into several sub-claims. Then, try to verify these sub-claims using only the facts I provide. Finally, give the final verification of the claim. The final verification should be either SUPPORTED or NOT_SUPPORTED. Please give the final verification in the last line. Here are some facts I provide: " + facts
            answer = prompt(system_input, user_input, model_name=model_name, show_message=True)
            ans = {}
            ans["answer"] = answer
            writer.write(json.dumps(ans))
            writer.write("\n")

def CoT_with_evidence_multi_round_test(result_path, few_shot=False):
    system_input = "You are a helpful assistant that verifies complex claims."
    user_input_prefix = "Please verify this claim: \""
    model_name = "gpt-3.5-turbo"
    few_shot_examples_with_prefix = 'Please decompose the claim into several sub-claims. If the claim do not need to be decomposed, just keep the original claim.' \
        + '\nHere are some examples for your reference: ' \
        + '\n\nclaim: "Arnold is currently the publisher and editorial director of Media Play News, one of five Hollywood trades and the only one dedicated to the home entertainment sector."' \
        + '\nsub-claims: ' \
        + '\n1. Arnold is currently the publisher and editorial director of Media Play News.' \
        + '\n2. Media Play News is one of five Hollywood trades.' \
        + '\n3. Media Play Newsis the only one dedicated to the home entertainment sector.' \
            \
        + '\n\nclaim: "Trump won the 2020 US Presidential Election."' \
        + '\nsub-claims:' \
        + '\n1. Trump won the 2020 US Presidential Election.' \
            \
        + '\n\nclaim: "Tazza  (TV series) is a 2008 South Korean television series starring the actor who played Prince Yeonsan in a film that runs 119 minutes."' \
        + '\nsub-claims:' \
        + '\n1. Tazza  (TV series) is a 2008 South Korean television series.' \
        + '\n2. One actor starred in Tazza also played Prince Yeonsan in a film.' \
        + '\n3. That film runs 119 minutes.' \
            \
        + '\n\nclaim: "Adam McKay co-wrote the film that Cassandra Lang made her cinematic debut in and served as head writer for "Saturday Night Live". ' \
        + '\nsub-claims:' \
        + '\n1. Adam McKay co-wrote the film that Cassandra Lang made her cinematic debut in.' \
        + '\n2. Adam McKay served as head writer for "Saturday Night Live".' \
            \
        + '\n\nclaim: '
    
    # few_shot_examples_with_prefix = 'Please decompose the claim into several sub-claims. If the claim do not need to be decomposed, just keep the original claim.' \
    #     + '\nHere is a example: ' \
    #     + '\n\nclaim: "Arnold is currently the publisher and editorial director of Media Play News, one of five Hollywood trades and the only one dedicated to the home entertainment sector."' \
    #     + '\nsub-claims: ' \
    #     + '\n1. Arnold is currently the publisher and editorial director of Media Play News.' \
    #     + '\n2. Media Play News is one of five Hollywood trades.' \
    #     + '\n3. Media Play Newsis the only one dedicated to the home entertainment sector.' \
    #     + '\n\nclaim: '
    with open(result_path, "w") as writer:
        for claim in tqdm(selected_dev_data[73:300]):
            facts = combine_facts(claim['facts'])
            if few_shot == True:
                decompose_input = few_shot_examples_with_prefix + claim['claim'].strip()
            else:
                decompose_input = user_input_prefix + claim['claim'].strip() + "\" First, you need to decompose the claim into several sub-claims:"
            # verify_input = "Then, try to verify these sub-claims using only the facts I provide. Finally, give the final verification of the claim. The final verification should be either SUPPORTED or NOT_SUPPORTED. Please give the final verification in the last line. Here are some facts I provide: " + facts
            verify_input = "Then, try to verify these sub-claims with explanations using only the facts I provide. "\
                + "The verification label must be either SUPPORTED or NOT_SUPPORTED. "\
                + "Finally, give the final verification label of the claim according to the verification of sub-claims. "\
                + "\nIf all sub-claims are supported by the facts, the final verification label should be SUPPORTED. "\
                + "\nIf any one of sub-claims is not supported, the final verification label should be NOT_SUPPORTED. "\
                + "\nHere are some facts I provide: \n" + facts
            user_input = [decompose_input, verify_input]
            history, answers = multi_round_prompt(system_input, user_input, model_name=model_name, show_message=True, few_shot=few_shot)
            ans = {}
            ans["decompose_answer"] = answers[0]
            ans["verify_answer"] = answers[1]
            ans["full_chat"] = history
            writer.write(json.dumps(ans))
            writer.write("\n")

'''
methods for evaluations
'''

def evaluate_direct(prediction_path, gold_path):
    r1 = open(prediction_path)
    r2 = open(gold_path)
    pred_data = [json.loads(line) for line in r1]
    gold_data = [json.loads(line) for line in r2]
    r1.close()
    r2.close()
    acc = 0
    wrong_id_list = []
    assert len(pred_data) == len(gold_data)
    for i in range(len(pred_data)):
        answer = pred_data[i]["answer"]
        if("NOT_SUPPORTED" in answer):
            short_answer = "NOT_SUPPORTED"
        elif ("SUPPORTED" in answer):
            short_answer = "SUPPORTED"
        else:
            short_answer = answer
            print("cannot find answer! id: %d; label: %s" % (i+1, short_answer.strip()))
        if short_answer == gold_data[i]["label"]:
            acc += 1
        else:
            wrong_id_list.append(i+1)
    print("accuracy:", acc/len(pred_data))
    return acc/len(pred_data), wrong_id_list

def evaluate_last_line(prediction_path, gold_path):
    r1 = open(prediction_path)
    r2 = open(gold_path)
    pred_data = [json.loads(line) for line in r1]
    gold_data = [json.loads(line) for line in r2]
    assert len(pred_data) == len(gold_data)
    acc = 0
    total = len(pred_data)
    no_ans_cnt = 0
    wrong_id_list = []
    for i in range(len(pred_data)):
        short_answer = pred_data[i]["answer"].split("\n")[-1]
        if("NOT_SUPPORTED" in short_answer):
            short_answer = "NOT_SUPPORTED"
        elif(" SUPPORTED" in short_answer):
            short_answer = "SUPPORTED"
        else:
            print("answer error: cannot find answer. id: %d; answer: %s" % (i+1, short_answer))
            no_ans_cnt += 1
        if(short_answer == gold_data[i]["label"]):
            acc += 1
        else:
            wrong_id_list.append(i+1)
    print("accuracy:",acc/total)
    print("no answer count:", no_ans_cnt)
    return acc/total, wrong_id_list
        
def evaluate_multi_round(prediction_path, gold_path):
    r1 = open(prediction_path)
    r2 = open(gold_path)
    pred_data = [json.loads(line) for line in r1]
    gold_data = [json.loads(line) for line in r2]
    # gold_data = gold_data[100:300]
    assert len(pred_data) == len(gold_data)
    acc = 0
    total = len(pred_data)
    no_ans_cnt = 0
    wrong_id_list = []
    for i in range(len(pred_data)):
        short_answer = pred_data[i]["verify_answer"].split("\n")[-1]
        # print(short_answer)
        if("NOT_SUPPORTED" in short_answer or "NOT SUPPORTED" in short_answer):
            short_answer = "NOT_SUPPORTED"
        elif("SUPPORTED" in short_answer):
            short_answer = "SUPPORTED"
        else:
            print("answer error: cannot find answer. id: %d; answer: %s" % (i+1, short_answer))
            no_ans_cnt += 1
            print(short_answer)
        if(short_answer == gold_data[i]["label"]):
            acc += 1
        else:
            wrong_id_list.append(i+1)
    print("accuracy:",acc/total)
    print("no answer count:", no_ans_cnt)
    return acc/total, wrong_id_list

def present_case(case_id, prediction_path, display_path):
    case_id -= 1
    r1 = open(prediction_path)
    pred_data = [json.loads(line) for line in r1]
    r1.close()
    claim = claim_list[case_id]
    pred_json = pred_data[case_id]
    evidence = selected_dev_data[case_id]['facts']
    label = selected_dev_data[case_id]['label']
    with open(display_path, "w") as writer:
        writer.write("Claim: " + claim + "\n")
        writer.write("Gold label: " + label + "\n")
        if("answer" in pred_json.keys()):
            writer.write("GPT-3.5's answer: \n" + pred_json['answer'] + "\n\n")
        elif("verify_answer" in pred_json.keys()):
            writer.write("GPT-3.5's decomposition: \n" + pred_json['decompose_answer'] + "\n\n")
            writer.write("GPT-3.5's verification: \n" + pred_json['verify_answer'] + "\n\n")
        writer.write("GOLD_EVIDENCE: \n")
        for item in evidence:
            writer.write("title: " + item['title'] + "\n")
            for item in item['text']:
                writer.write(item + "\n")
            writer.write("\n")
    




def final_evaluate(mode):
    assert mode in ["direct", "direct_CoT", "direct_CoT_with_evidence", "direct_with_evidence", "CoT", "CoT_with_evidence", "CoT_multi_round", "CoT_multi_round_oneshot", "CoT_multi_round_fewshot"]
    if(mode == "direct"):
        pred_file = "results/direct/dev.jsonl"
    elif(mode == "direct_CoT"):
        pred_file = "results/direct/dev_CoT.jsonl"
    elif(mode == "direct_with_evidence"):
        pred_file = "results/direct/dev_with_evidence.jsonl"
    elif(mode == "CoT"):
        pred_file = "results/CoT/dev_naive_CoT_updat_2.jsonl"
    elif(mode == "CoT_with_evidence"):
        pred_file = "results/CoT/dev_naive_CoT_with_evidence.jsonl"
    elif(mode == "CoT_multi_round"):
        pred_file = "results/CoT/dev_CoT_with_evidence_multi_round.jsonl"
    elif(mode == "CoT_multi_round_oneshot"):
        pred_file = "results/CoT/dev_CoT_with_evidence_multi_round_oneshot.jsonl"
    elif(mode == "CoT_multi_round_fewshot"):
        pred_file = "results/CoT/dev_CoT_with_evidence_multi_round_fewshot.jsonl"
    elif(mode == "direct_CoT_with_evidence"):
        pred_file = "results/direct/dev_with_evidence_LTSBS.jsonl"
    else:
        print(mode)
        raise ValueError("mode error")
    
    pred_data = [json.loads(line) for line in open(pred_file, "r")]
    gold_data = [json.loads(line) for line in open("dev_selected.json", "r")]
    selected_pred_data = []
    selected_gold_data = []
    for i in range(0, 95):
        if i+1 not in [9,14,22,35,38,56,59,61,63,67,68,71,75,77,86]:
            selected_pred_data.append(pred_data[i])
            selected_gold_data.append(gold_data[i])
    assert len(selected_pred_data) == 80
    
    for i in range(250, 299):
        if i+1 not in [260,255,264,265,269,277,279,284,293]:
            selected_pred_data.append(pred_data[i])
            selected_gold_data.append(gold_data[i])
    assert len(selected_pred_data) == 120
    assert len(selected_gold_data) == 120
    
    acc = 0
    total = len(selected_pred_data)
    no_ans_cnt = 0
    wrong_id_list = []
    TP=0
    FP=0
    FN=0
    for i in range(total):
        if mode in ["direct", "direct_with_evidence"]:
            short_answer = selected_pred_data[i]["answer"][:15]
        elif mode in ["direct_CoT", "CoT", "CoT_with_evidence", "direct_CoT_with_evidence"]:
            short_answer = selected_pred_data[i]["answer"].split("\n")[-1]
        else:
            short_answer = selected_pred_data[i]["verify_answer"].split("\n")[-1]
        # print(short_answer)
        if("NOT_SUPPORTED" in short_answer or "NOT SUPPORTED" in short_answer):
            short_answer = "NOT_SUPPORTED"
        elif("SUPPORTED" in short_answer):
            short_answer = "SUPPORTED"
        else:
            print("answer error: cannot find answer. id: %d; answer: %s" % (i+1, short_answer))
            no_ans_cnt += 1
            print(short_answer)
        if(short_answer == selected_gold_data[i]["label"]):
            acc += 1
            if(short_answer == "SUPPORTED"):
                TP += 1
        else:
            if(short_answer == "SUPPORTED"):
                FP += 1
            else:
                FN += 1
            wrong_id_list.append(i+1)
    # print("accuracy:",acc/total)
    print("no answer count:", no_ans_cnt)
    print("TP:", TP, "FP:", FP, "FN:", FN)
    return acc/total, TP/(TP+FP), TP/(TP+FN), wrong_id_list
            
            

    
    
if __name__ == "__main__":
    # direct_test("dev_direct_update.jsonl")
    # direct_CoT_test("dev_direct_CoT.jsonl")
    # naive_CoT_test("dev_naive_CoT_updat_2.jsonl")
    # direct_with_evidence_test("results/direct/dev_with_evidence_LTSBS.jsonl")
    # CoT_with_evidence_test("results/CoT/dev_naive_CoT_with_evidence.jsonl")
    # CoT_with_evidence_multi_round_test("results/CoT/dev_CoT_with_evidence_multi_round_fewshot_tmp.jsonl", few_shot=True)
    # acc, wrong_id_list = evaluate_multi_round("results/CoT/dev_CoT_with_evidence_multi_round_oneshot.jsonl", "dev_selected.json")
    # acc, wrong_ixd_list = evaluate_multi_round("results/CoT/dev_CoT_with_evidence_multi_round.jsonl", "dev_selected.json")
    # # acc, wrong_id_list = evaluate_direct("results/direct/dev_direct.jsonl", "dev_selected.json")
    # acc, wrong_id_list = evaluate_last_line("results/CoT/dev_naive_CoT_with_evidence.jsonl", "dev_selected.json")
    # acc, wrong_id_list = evaluate_direct("dev_with_evidence.jsonl", "dev_selected.json")
    # with open("results/CoT/dev_CoT_with_evidence_multi_round_wrong_id.txt", "w") as writer:
    #     writer.write("这是dev_CoT_with_evi_multi_round_wrong_id_2的错误id列表\n")
    #     for id in wrong_id_list:
    #         writer.write(str(id+100))
    #         writer.write("\n")
    # present_case(275, "results/CoT/dev_CoT_with_evidence_multi_round_fewshot.jsonl", "display_wrong_cases.txt")
    # present_case(301, "results/CoT/dev_CoT_with_evidence_multi_round_oneshot.jsonl", "display_wrong_cases.txt")
    modes = ["direct", "direct_CoT", "direct_with_evidence", "direct_CoT_with_evidence", "CoT", "CoT_with_evidence", "CoT_multi_round", "CoT_multi_round_oneshot", "CoT_multi_round_fewshot"]
    for mode in modes:
        # mode = "CoT_multi_round_fewshot"
        acc, precision, recall, wrong_id_list = final_evaluate(mode)
        print("mode: %s, acc: %f, recall: %f, precision: %f" % (mode, acc, recall, precision))
        print("wrong_id_list", wrong_id_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def evaluate_naive_CoT(prediction_path, gold_path):
    #     r1 = open(prediction_path)
#     r2 = open(gold_path)
#     pred_data = [json.loads(line) for line in r1]
#     gold_data = [json.loads(line) for line in r2]
#     r1.close()
#     r2.close()
#     acc = 0
#     assert len(pred_data) == len(gold_data)
#     for i in range(len(pred_data)):
#         pred = pred_data[i]['answer'].lower()
#         if("overall claim" in pred):
#             final_prediction = pred.split("overall claim")[-1].strip()
#         elif("original claim" in pred):
#             final_prediction = pred.split("original claim")[-1].strip()
#         else:
#             last_support = pred.rfind(" supported") # note that there is a space before supported, which is important
#             last_not_support = pred.rfind("not_supported")
#             if last_support == -1 and last_not_support == -1:
#                 print("pred:", pred)
#                 print("cannot find the final prediction! id:", i+1)
#             elif(last_support > last_not_support):
#                 final_prediction = "supported"
#             else:
#                 final_prediction = "not_supported"
#         # print("final_prediction", final_prediction)
#         if "supported" in final_prediction and gold_data[i]['label'] == "SUPPORTED":
#             acc += 1
#         elif "not_supported" in final_prediction and gold_data[i]['label'] == "NOT_SUPPORTED":
#             acc += 1
#         else:
#             print("wrong id", i+1)      
#     print("accuracy:", acc/len(pred_data))
#     return acc/len(pred_data)