import json
answer_list = []
i = 0
with open("results/FlanT5-large/HoVer_dev_all.txt", "r") as reader:
    writer = open("results/FlanT5-large/dev_selected_predictions.txt", "w")
    for item in reader:
        if (i % 4) == 0:
            answer_list.append(item)
            writer.write(item)
        i += 1
    writer.close() # 一定要记得close()，否则写入的内容可能一部分还在缓冲区里，没有写入文件！！


pred_data = answer_list
gold_data = [json.loads(line) for line in open("dev_selected.json", "r")]
selected_pred_data = []
selected_gold_data = []
for i in range(0, 95):
    if i+1 not in [9,14,22,35,38,56,59,61,63,67,68,71,75,77,86]:
        selected_pred_data.append(pred_data[i].split("\n")[0])
        selected_gold_data.append(gold_data[i])
assert len(selected_pred_data) == 80

for i in range(250, 299):
    if i+1 not in [260,255,264,265,269,277,279,284,293]:
        selected_pred_data.append(pred_data[i].split("\n")[0])
        selected_gold_data.append(gold_data[i])
assert len(selected_pred_data) == 120
assert len(selected_gold_data) == 120

acc = 0
TP=0
FP=0
FN=0

for i in range(len(selected_pred_data)):
    if selected_gold_data[i]["label"] == "SUPPORTED" and selected_pred_data[i] == "SUPPORTS":
        acc += 1
        TP += 1
    elif selected_gold_data[i]["label"] == "NOT_SUPPORTED" and selected_pred_data[i] == "REFUTES":
        acc += 1
    elif selected_gold_data[i]["label"] == "SUPPORTED" and selected_pred_data[i] == "REFUTES":
        FN += 1
    else:
        FP += 1
        
print("TP:", TP, "FP:", FP, "FN:", FN)
total = 120
print(acc/total, TP/(TP+FP), TP/(TP+FN))