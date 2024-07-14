---
library_name: transformers
tags:
- pytorch
license: apache-2.0
language:
- ko
pipeline_tag: text-generation
---

<p align="left">
  <img src="https://huggingface.co/cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0/resolve/main/ocelot.webp" width="50%"/>
<p>

# solar-kor-resume

> Update @ 2024.05.27: First release of Ocelot-Ko-self-instruction-10.8B-v1.0
<!-- Provide a quick summary of what the model is/does. -->

This model card corresponds to the 10.8B Instruct version of the **Solar-Ko** model. 

The train wad done on A100-80GB

**Resources and Technical Documentation**:
* [Solar Model](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)


**Citation**

```bibtex
@misc {cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0,
	author       = { {frcp, nebchi, pepperonipizza97} },
	title        = { solar-kor-resume},
	year         = 2024,
	url          = { https://huggingface.co/cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0 },
	publisher    = { Hugging Face }
}
```

**Model Developers**: frcp, nebchi, pepperonipizza97

## Model Information

Resume Proofreading and evaluation of inputs and outputs.

### Description
It has been trained with a large amount of Korean tokens compared to other LLMs, enabling it to generate high-quality Korean text. 

**Model Architecture** Solar is an auto-regressive language model that is scaled using the DUS method. 

*You can find dataset list here: https://huggingface.co/datasets/cpm-ai/gpt-self-introduction-all

### Inputs and outputs
*   **Input:** Text string, such as a question, a prompt, or a document to be
    Proofreaded.
*   **Output:** Generated Korea text in response to the input, such
    as an answer to a question, or a evaluation of a resume.

#### Running the model on a single / multi GPU
```python
# pip install accelerate, flash_attn, sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0")
model = AutoModelForCausalLM.from_pretrained("cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0", device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096, streamer=streamer)

text = 너는 자기소개서 첨삭 전문가야.
주어진 자기소개서를 첨삭해서 다시 작성해야해.
출력형식은 다음을 지켜야해.

[첨삭]

다음이 자기소개서야 :
[저는 어린 시절부터 완벽주의적인 성격을 가지고 있었습니다. 이로 인해 항상 자신의 능력에 대한 불안감을 느끼며 과도한 스트레스를 받아왔습니다. 학창 시절에는 과제나 프로젝트를 완벽하게 마무리하지 못하면 자존감이 크게 흔들렸습니다. 중학교 시절에는 한 가지 문제에 너무 오랜 시간을 투자하여 다른 학습 기회를 놓치기도 했습니다. 이러한 경험들은 저에게 완벽함을 추구하는 것이 종종 현실에 부적합하다는 것을 깨달게 했습니다.

고등학교와 대학교에 진학하면서도 이러한 완벽주의적인 성격을 극복하기 위해 노력했습니다. 하지만 여전히 실패를 받아들이는 것이 어렵고, 자신의 한계를 인정하는 것이 어려웠습니다. 이러한 과정을 통해 완벽함에 대한 강박이 저의 성장과정을 제약하는 요인이 되었음을 깨달았습니다.]"""

messages = [
    {
        "role": "user",
        "content": "{}".format(text)
    }
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(
    prompt,
    temperature=0.2,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])

```

### results

```python
[첨삭]

어린 시절부터 저는 완벽한 결과를 추구하며 스스로를 압박해온 성격이었습니다. 이는 학업과 관련된 스트레스로 이어져, 과제를 완수하더라도 만족보다는 불만족의 감정이 더 컸던 시기였습니다. 특히 중학교 때 한 문제를 해결하기 위해 지나치게 오랫동안 매달려 결국 중요한 시기를 놓친 경험은 제 성장에 큰 영향을 미쳤습니다. 이 과정에서 완벽주의를 추구하는 것이 현실적이지 않을 수 있다는 사실을 깨닫기 시작했습니다.

고등학교와 대학에서는 이러한 성향을 개선하고자 다양한 노력을 기울였습니다. 예를 들어, 목표를 세분화하고 단계별로 접근하면서 성취감과 자신감을 키우기 위해 노력했습니다. 또한, 팀 프로젝트에서 역할을 분담하고 협력함으로써 개인의 한계보다 전체 성과를 우선시하는 법을 배웠습니다. 비록 아직 완벽함이라는 굴레로부터 완전히 자유로워지지는 못했지만, 이를 극복하고 성장할 수 있는 방법을 찾았다는 점에서 자부심을 느낍니다.
```
                                                                    
### Evaluation Results - LogicKor
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Evaluation Results-LogicKor</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: left;
        }
        th, td {
            padding: 12px 15px;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-of-type(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
</head>
<body>
    <table border="1">
        <thead>
            <tr>
                <th>Model</th>
                <th>글쓰기</th>
                <th>이해</th>
                <th>문법</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>HyperClovaX</td>
                <td>8.50</td>
                <td>9.50</td>
                <td><b>8.50</b></td>
            </tr>
            <tr>
                <td>solar-1-mini-chat</td>
                <td>8.50</td>
                <td>7.00</td>
                <td>5.21</td>
            </tr>         
            <tr>
                <td>allganize/Llama-3-Alpha-Ko-8B-Instruct</td>
                <td>8.50</td>
                <td>8.35</td>
                <td>4.92</td>
            </tr>
            <tr>
                <td>Synatra-kiqu-7B</td>
                <td>4.42</td>
                <td>5.71</td>
                <td>4.50</td>
            </tr>
            <tr>
                <td><b>Ocelot-ko-10.8B</b></td>
                <td><b>8.57</b></td>
                <td>7.00</td>
                <td>6.57</td>
            </tr>
        </tbody>
    </table>
</body>
</html>

### Evaluation Results - Kobest
| 모델 명칭          |**Average**<br>n=0&nbsp;n=5  |HellaSwag<br>n=0&nbsp;&nbsp;n=5 |COPA<br> n=0&nbsp;&nbsp;n=5 |BooIQ<br>n=0&nbsp;&nbsp;n=5 | 
|------------------ |------------------------------|------------------------------|------------------------------|------------------------------|
| KoGPT             |  58.2   &nbsp;&nbsp;   63.7   |  55.9   &nbsp;&nbsp;   58.3   |  73.5   &nbsp;&nbsp;   72.9   |  45.1   &nbsp;&nbsp;   59.8  | 
| Polyglot-ko-13B   |  62.4   &nbsp;&nbsp;   68.2   |**59.5** &nbsp;&nbsp; **63.1** |**79.4** &nbsp;&nbsp;   81.1   |  48.2   &nbsp;&nbsp;   60.4  |  
| LLaMA 2-13B       |  45.2   &nbsp;&nbsp;   60.5   |  41.3   &nbsp;&nbsp;   44.0   |  59.3   &nbsp;&nbsp;   63.8   |  34.9   &nbsp;&nbsp;   73.8  | 
| Baichuan 2-13B    |  52.7   &nbsp;&nbsp;   53.9   |  39.2   &nbsp;&nbsp;   39.6   |  60.6   &nbsp;&nbsp;   60.6   |  58.4   &nbsp;&nbsp;   61.5  | 
| QWEN-14B          |  47.8   &nbsp;&nbsp;   66.4   |  45.3   &nbsp;&nbsp;   46.8   |  64.9   &nbsp;&nbsp;   68.9   |  33.4   &nbsp;&nbsp;   83.5  | 
| Orion-14B-Chat    |  68.8   &nbsp;&nbsp;   73.2   |  47.0   &nbsp;&nbsp;   49.6   |  77.7   &nbsp;&nbsp;   79.4   |  81.6   &nbsp;&nbsp;   90.7  |                                                                               
| **Ocelot-ko-10.8B**   |**72.5** &nbsp;&nbsp; **75.9** |  50.0   &nbsp;&nbsp;   51.4   |  75.8   &nbsp;&nbsp; **82.5** |**91.7** &nbsp;&nbsp; **93.8**|  

### Software
Training was done using QLoRA
---