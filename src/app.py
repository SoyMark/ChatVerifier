import gradio as gr
import openai
import sys
import time
import requests
import urllib.parse
import re
import threading
from tqdm import tqdm
import json


# def prepara_data(dev_file_path):
#     selected_dev_data = []
#     cnt = 0
#     with open(dev_file_path, "r") as reader:
#         dev_data = json.load(reader)
#         for i in range(len(dev_data)):
#             if (i % 4) == 0:
#                 cnt += 1
#                 js = {}
#                 js["id"] = str(cnt)
#                 js["claim"] = dev_data[i]["claim"]
#                 js["label"] = dev_data[i]["label"]
#                 js["facts"] = dev_data[i]["supporting_facts_contain_text"]
#                 selected_dev_data.append(js)
#     claim_list = [d["claim"] for d in selected_dev_data]
#     return selected_dev_data, claim_list

# selected_dev_data, claim_list = prepara_data("dev.json")


class MetaAPI:
    def __init__(self, api_key, api_name: str, base_url: str, proxy=None):
        self.proxy = proxy
        self.session = requests.Session()
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.api_key = api_key
        self.api_name = api_name
        self.base_url = base_url
        self.lock = threading.Lock()
        
class WikiSearchAPI(MetaAPI):
    def __init__(self, proxy=None):
        api_name = 'Wiki Search'
        base_url = 'https://en.wikipedia.org/w/api.php?'
        super(WikiSearchAPI, self).__init__(api_name=api_name, base_url=base_url, api_key=None, proxy=proxy)
    def call(self, query, num_results=4):
        def remove_html_tags(text):
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
        }
        call_url = self.base_url + urllib.parse.urlencode(params)
        r = self.session.get(call_url)
        data = r.json()['query']['search']
        data = [d['title'] + ": " + remove_html_tags(d["snippet"]) for d in data][:num_results]
        return data

class ChatVerifier:
    def __init__(self,):
        self.search_engine = WikiSearchAPI()
        
    def _prompt_generator(self, claim: str, round: int, evidence='', few_shot=True):
        system_input = "You are a helpful assistant that decompose complex claims. If the claim do not need to be decomposed, please say BRIEF."
        user_input_prefix = "Please verify this claim no matter if it is true or not: \""
        claim = claim.strip()
        if claim[-1] != '.':
            claim = claim + '.'
        assert round in [1, 2]
        if round == 1:
            few_shot_examples_with_prefix = 'Please decompose the claim into several sub-claims, no matter if it is true.' \
                + 'If the claim can not be decomposed, just say BRIEF.' \
                + '\nHere are some examples: ' \
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
        
            # few_shot_examples_with_prefix = 'Please decompose the claim into several sub-claims, no matter if it is true.' \
            #     + 'If the claim can not be decomposed, just say BRIEF.' \
            #     + '\nHere is a example: ' \
            #     + '\n\nclaim: "Arnold is currently the publisher and editorial director of Media Play News, one of five Hollywood trades and the only one dedicated to the home entertainment sector."' \
            #     + '\nsub-claims: ' \
            #     + '\n1. Arnold is currently the publisher and editorial director of Media Play News.' \
            #     + '\n2. Media Play News is one of five Hollywood trades.' \
            #     + '\n3. Media Play Newsis the only one dedicated to the home entertainment sector.' \
            #     + '\n\nclaim: '
            if few_shot == True:
                decompose_input = few_shot_examples_with_prefix + claim.strip()
            else:
                decompose_input = user_input_prefix + claim.strip() + "\" First, you need to decompose the claim into several sub-claims:"
            user_input = decompose_input
        elif round == 2:
            facts = ".\n".join(evidence) + ". "
            verify_input = "Then, try to verify these sub-claims with explanations using only the facts I provide. "\
                + "The verification label should only be either SUPPORTED or REFUTED or NOT_ENOUGH_EVIDENCE. "\
                + "Finally, give the final verification label of the claim according to sub-claims. "\
                + "\nIf all sub-claims are supported, the final verification label should be SUPPORTED. "\
                + "\nIf any one of sub-claims is verified as REFUTED, the final verification label should be REFUTED. "\
                + "\nIn other cases, the final verification label should be NOT_ENOUGH_EVIDENCE. "\
                + "\nHere are some facts I provide: \n" + facts
            user_input = verify_input
        return system_input, user_input
        
    def _prompt_gpt(self, user_input, system_input='', history=[], model_name='gpt-3.5-turbo', show_message=False):
        user_json = {"role": "user", "content": user_input}
        if history == []:
            assert system_input != ''
            system_json = {"role": "system", "content": system_input}
            message = [system_json, user_json]
            history = message
        else:
            message = history
            message.append(user_json)
        for _ in range(3): # 尝试至多3次
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=message,
                    temperature=0.3,
                    max_tokens=1000,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                time.sleep(1)
        answer = response['choices'][0]['message']['content']
        history.append(response['choices'][0]['message'])
        if show_message == True: 
            print("chat history:", history)
        return answer, history
    
    def _gather_evidence(self, orginal_claim, decompose_answer):
        if ("BRIEF" in decompose_answer):
            sub_claims = [orginal_claim]
        else:
            sub_claims = decompose_answer.split("\n")[1:]
            sub_claims = [item[2:].strip() for item in sub_claims]
        evidence = []
        for c in sub_claims:
            evidence += self.search_engine.call(c)
        return evidence

    def multi_round_verify(self, model_name, claim, few_shot=True):
        # first round: decompose
        round = 1
        system_input, user_input = self._prompt_generator(claim, round)
        decompose_answer, history = self._prompt_gpt(user_input, system_input=system_input, model_name=model_name, show_message=True)
        if ("BRIEF" in decompose_answer): # no need to decompose
            for item in history:
                if(item['role'] == 'assistant'):
                    item['content'] = f"sub claims:\n1.{claim}"
            decompose_answer = f"sub claims:\n1.{claim}"
        # second round: retrieve and verify
        round = 2
        #revise system input
        for item in history:
            if(item['role'] == 'system'):
                item['content'] = "You are a helpful assistant that verify complex claims"
            if(item['role'] == 'user'):
                item['content'] = "Please decompose the claim into several sub-claims, no matter if it is true.\n"\
                    + "Claim: " + claim 
        evidence = self._gather_evidence(claim, decompose_answer)
        _, user_input = self._prompt_generator(claim, round, evidence=evidence, few_shot=few_shot)
        verify_answer, history = self._prompt_gpt(user_input, history=history, model_name=model_name, show_message=True)
        brief_answer = ""
        index = verify_answer.lower().find("final verification")
        if index != -1:
            final_verification = verify_answer[index:]
        else:
            print("Warning: final verification not found")
            final_verification = verify_answer
        for key_word in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE"]:
            if key_word in final_verification:
                brief_answer = key_word
                break
        if brief_answer == "": 
            brief_answer = verify_answer
        
        formatted_evidence = ""
        for evi in evidence:
            formatted_evidence += evi + "\n\n"
        return brief_answer, decompose_answer, verify_answer, formatted_evidence
    
    def verify(self, model_name, claim, debug=False):
        evidence = self.search_engine.call(claim)
        system_input, user_input = self._prompt_generator(claim, evidence)
        answer = self._test(system_input, user_input, model_name=model_name, show_message=debug)
        brief_answer = answer.split("\n")[-1].split(".")[0]
        return brief_answer, answer, evidence
    
if __name__ == "__main__":
    
    input_text = gr.inputs.Textbox(label="Please type in a claim")
    output_text = [gr.outputs.Textbox(label="Brief answer from Chatgpt"), gr.outputs.Textbox(label="Detailed answer from Chatgpt"), gr.outputs.Textbox(label="Retrieved evidence from WikiSearchAPI")]
    title = "<div align='center'><h1>Chatgpt for fact verification</h1></div>"
    checker = ChatVerifier()
    iface = gr.Interface(
        fn=checker.multi_round_verify, 
        inputs=[
            gr.Dropdown(['gpt-4', 'gpt-3.5-turbo'], value='gpt-3.5-turbo', label='模型'), # model_name
            gr.Textbox(label="待验证的复杂事实"), # input_prompt
        ],
        outputs=[gr.Textbox(label='Brief answer'), gr.outputs.Textbox(label="Decompositions"), gr.outputs.Textbox(label="Detailed answer"), gr.outputs.Textbox(label="Retrieved evidences from WikiSearchAPI")],
        title="ChatVerifier: 复杂事实检验器",
        description="<div align='center'>基于大模型的英文复杂事实检验。使用方法：输入一句你想要检验的陈述，模型会自动对你陈述进行验证，并提供解释。</div>",
        allow_flagging='manual',
        flagging_dir='flagged/',
        flagging_options=['👍', '👎'],
        examples=[
            ['gpt-3.5-turbo', 'Fuling District and Qidong, Jiangsu are both located in China.'], 
            ['gpt-3.5-turbo', 'The movie adaption of the novel "In Her Shoes" was released in 2005. It was written by the author of Good In Bed.'],
            ['gpt-3.5-turbo', 'At the awards ceremony where Rudaali was selected as an entry for the Best Foreign Language Film but, was not accepted as a nominee, the awards for technical achievements were presented by host Laura Dern who was nominated for Best Actress for "Rambling Rose".']
        ],
    )
    iface.queue(concurrency_count=3).launch(share=True)
    iface.close()
    
    # def test_app(result_path):
    #     verifier = ChatVerifier()
    #     with open(result_path, "w") as writer:
    #         for claim in tqdm(selected_dev_data[:400]):
    #             brief_answer, verify_answer, evidence = verifier.multi_round_verify("gpt-3.5-turbo", claim['claim'])
    #             writer.write(json.dumps({"claim": claim['claim'], "brief_answer": brief_answer, "verify_answer": verify_answer, "evidence": evidence}) + "\n")
    # test_app("results/CoT/dev_CoT_app.jsonl")