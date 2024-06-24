# Awesome-SLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Awesome-SLM: a curated list of Small Language Model

![](resources/image1.png)

🔥 Small Language Models(SLM) are streamlined versions of large language models designed to retain much of the original capabilities while being more efficient and manageable. Here is a curated list of papers about small language models. 


## Table of Content

- [Awesome-SLM ](#awesome-slm-)
  - [Milestone Papers](#milestone-papers)
  - [Other Papers](#other-papers)
  - [SLM Leaderboard](#slm-leaderboard)
  - [Open SLM](#open-slm)
  - [SLM Data](#slm-data)
  - [SLM Evaluation](#slm-evaluation)
  - [Miscellaneous](#miscellaneous)

## Milestone Papers

|  Date  |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     Transformers     |      Google      | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       GPT 1.0       |      OpenAI      | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2018-10 |         BERT         |      Google      | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf2b0e26d0599ce3e70df8a9da02e51594e0e992%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |
| 2019-10 |          T5          |      Google      | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)                                                           |    JMLR<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3cfb319689f06bf04c2e28399361f414ca32c4b3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)  |
| 2020-01 |     Scaling Law     |      OpenAI      | [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)                                                                                                        |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe6c561d02500b2596a230b341a8eb8b921ca5bf2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2021-09 |         FLAN         |      Google      | [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR)                                                                                        |    ICLR <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fff0b2681d7b05e16c46dfb71d980cc2f605907cd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2021-10 |         T0         |      HuggingFace et al.      | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)                                                                                        |    ICLR <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F17dd3555fd1ccf1141cf984347fa1b3fd6b009ca%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2022-01 |        LaMDA        |      Google      | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)                                                                                                 |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb3848d32f7294ec708627897833c4097eb4d8778%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2022-06 |  Emergent Abilities  |      Google      | [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)                                                                                                |    TMLR<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdac3a172b504f4e33c029655e9befb3386e5f63a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2023-02 | LLaMA | Meta |[LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)|![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F57e849d0de13ed5f91d086936296721d4ff75a75%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2023-03 | Alpaca | Stanford |[Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)| No paper |
| 2023-06 | Orca | Microsoft | [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/pdf/2306.02707) |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0244aeb7c6927e2fb0c2e668687e160a00737dbe%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2023-07 | LLaMA 2 | Meta | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf) |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F104b0bb1da562d53cbda87aec79ef6a2827d191a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 2023-07 | Stable Beluga | stability.ai | [Meet Stable Beluga 1 and Stable Beluga 2, Our Large and Mighty Instruction Fine-Tuned Language Models](https://stability.ai/news/stable-beluga-large-instruction-fine-tuned-models) | No paper |
|2023-09| Xgen 7B | Salesforce Research |[XGen-7B Technical Report](https://arxiv.org/pdf/2309.03450)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd9d84f8b9e6bf4839f43ab865158d79a3b9bb1e9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-09| Qwen | Alibaba |[QWEN TECHNICAL REPORT](https://arxiv.org/pdf/2309.16609)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5fc1a3a49e8f1d106118b69d1d6be3b6caa23da0%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-10| Mistral 7B| Mistral |[Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdb633c6b1c286c0386f0078d8a2e6224e03a6227%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-10| Zephyr 7B|  HuggingFace et al. |[Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/pdf/2310.16944)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcdcf3f36866ef1e16eba26d57c2324362247ba84%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-10| MBT 7B| Mosaic Research |[Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.databricks.com/blog/mpt-7b)| No paper |
|2023-11| Falcon 7B|  Technology Innovation Institute |[The Falcon Series of Open Language Models](https://arxiv.org/pdf/2311.16867)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2c0312c604f9f7638bb4533b39e0ae81e7f6ab12%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-11| Orca 2 |  Microsoft |[Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/pdf/2311.11045)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd2ea161e6b2114529d875d16cfaad4c824e17a8c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2023-12| Phi-2 |  Microsoft |[Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)| No paper |
|2024-01| TinyLlama 1.1B | StatNLP Group |[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385.pdf)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F560c6f24c335c2dd27be0cfa50dbdbb50a9e4bfd%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-01| LLaVA-Phi | Midea Group |[LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model](https://arxiv.org/pdf/2401.02330.pdf)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fece33ee67d74c29cd2a83c505e5bf0b818f9c2a1%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-01| H2O-Danube 1.8B | H2O.ai |[H2O-Danube-1.8B Technical Report](https://arxiv.org/pdf/2401.16818)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F139f3e52d1b90992d6908497ed8d6535bad258a2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-01| TeleChat | chinatelecom |[TeleChat Technical Report](https://arxiv.org/pdf/2401.03804)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F400e25e16276404097889e30b1d53e227ad30b83%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-02| Nemotron-4 15B | NVIDIA |[Nemotron-4 15B Technical Report](https://arxiv.org/pdf/2402.16819)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb54c3599a17db5a71349877a8567400117efbade%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-03| Yi | HuggingFace et al. |[Yi: Open Foundation Models by 01.AI](https://arxiv.org/pdf/2403.04652)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc0b454e0a6aa51ff3ba56778787d0c43932ef6ba%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-03| Gemma | Google DeepMind |[Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/pdf/2403.08295)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb842b83a7ff5dff8e3b83915d8c15423b6085728%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-03| Jamba | AI21labs |[Jamba:A Hybrid Transformer-Mamba Language Model](https://arxiv.org/pdf/2403.19887)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcbaf689fd9ea9bc939510019d90535d6249b3367%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-04|  CT-LLM 2B | Multimodal Art Projection Research Community |[Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model](https://arxiv.org/pdf/2404.04167)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3e9141595ab2820be0b461c9ca61ed86aa9a5b15%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-04|  RHO-1 | Microsoft |[RHO-1: Not All Tokens Are What You Need](https://arxiv.org/pdf/2404.07965)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcafe794035266eaef5d53e5d37aa486c71db9703%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-04|  Phi-3 | Microsoft |[Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fabdceff7d7983cdede9a5aabe6a476d4c72e41a3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-05| Zamba 7B | Zyphra |[Zamba: A Compact 7B SSM Hybrid Model](https://arxiv.org/pdf/2405.16712)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3dd6bc488283f1e4cc967d98a6a6c3d7f1a6cf76%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-05| ChuXin 1.6B | HuggingFace et al. |[ChuXin: 1.6B Technical Report](https://arxiv.org/pdf/2405.04828)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2a9c93165f805767bf5beafb985b8320f6cfdb15%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-05| OpenBA-V2 3.4B | Soochow University |[OpenBA-V2: Reaching 77.3% High Compression Ratio with Fast Multi-Stage Pruning](https://arxiv.org/pdf/2405.05957)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F59768c67974ee3d75b427ee5d1af2095e49fc1ea%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-05| Aya 23 | Cohere |[Aya 23: Open Weight Releases to Further Multilingual Progress](https://arxiv.org/pdf/2405.15032)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff1549691db69b6564a0b2a52c795a91d4a5a2c71%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-06| Xmodel-LM 1.1B | XiaoduoAI |[Xmodel-LM Technical Report](https://arxiv.org/pdf/2406.02856)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff4b9ccba1813143347056a1cb0e37e946a7d19b2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-06| SAMBA 3.8B | microsoft |[SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/pdf/2406.07522)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28eb18717cfa257f0fc49fb9512c48279cafa031%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|




## Other Papers
If you're interested in the field of SLM, you may find the above list of milestone papers helpful to explore its history and state-of-the-art. However, each direction of SLM offers a unique set of insights and contributions, which are essential to understanding the field as a whole. For a detailed list of papers in various subfields, please refer to the following link:

|2024-02| Ensemble SLMs | Nanyang Technological University |[Purifying Large Language Models by Ensembling a Small Language Model](https://arxiv.org/pdf/2402.14845)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8287fbf2c520f7604aae19a590ea9494d356aae7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
|2024-01| Vary-toy 1.8B | MEGVII Technology |[Small Language Model Meets with Reinforced Vision Vocabulary](https://arxiv.org/pdf/2401.12503.pdf)|<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F41f12456780aecd204a210ce04b1a92d022b8c4c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|

## SLM Leaderboard
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) - a benchmark platform for large language models (LLMs) that features anonymous, randomized battles in a crowdsourced manner.
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) - An Automatic Evaluator for Instruction-following Language Models 
 using Nous benchmark suite. 
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - aims to track, rank and evaluate LLMs and chatbots as they are released.
- [OpenCompass 2.0 LLM Leaderboard](https://rank.opencompass.org.cn/leaderboard-llm-v2) - OpenCompass is an LLM evaluation platform, supporting a wide range of models (InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.

## Open SLM
- Meta
  - [Llama 2-7|13|70B](https://llama.meta.com/llama2/)
  - [Llama 1-7|13|33|65B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
  - [OPT-1.3|6.7|13|30|66B](https://arxiv.org/abs/2205.01068)
- Mistral AI
  - [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
- Google
  - [Gemma-2|7B](https://blog.google/technology/developers/gemma-open-models/)
  - [RecurrentGemma-2B](https://github.com/google-deepmind/recurrentgemma)
  - [T5](https://arxiv.org/abs/1910.10683)
- Apple
  - [OpenELM-1.1|3B](https://huggingface.co/apple/OpenELM)
- Microsoft
  - [Phi1-1.3B](https://huggingface.co/microsoft/phi-1)
  - [Phi2-2.7B](https://huggingface.co/microsoft/phi-2)
  - [Phi3-3.8|7|14B](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- AllenAI
  - [OLMo-7B](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- xAI
  - [Grok-1-314B-MoE](https://x.ai/blog/grok-os)
- DeepSeek
  - [DeepSeek-Math-7B](https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
  - [DeepSeek-Coder-1.3|6.7|7|33B](https://huggingface.co/collections/deepseek-ai/deepseek-coder-65f295d7d8a0a29fe39b4ec4)
  - [DeepSeek-VL-1.3|7B](https://huggingface.co/collections/deepseek-ai/deepseek-vl-65f295948133d9cf92b706d3)
  - [DeepSeek-MoE-16B](https://huggingface.co/collections/deepseek-ai/deepseek-moe-65f29679f5cf26fe063686bf)
- Alibaba
  - [Qwen-1.8|7|14|72B](https://huggingface.co/collections/Qwen/qwen-65c0e50c3f1ab89cb8704144)
  - [Qwen1.5-1.8|4|7|14|32|72|110B](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)
  - [CodeQwen-7B](https://huggingface.co/Qwen/CodeQwen1.5-7B)
  - [Qwen-VL-7B](https://huggingface.co/Qwen/Qwen-VL)
  - [Qwen2-0.5|1.5|7|57-MOE|72B](https://qwenlm.github.io/blog/qwen2/)
- 01-ai
  - [Yi-34B](https://huggingface.co/collections/01-ai/yi-2023-11-663f3f19119ff712e176720f)
  - [Yi1.5-6|9|34B](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)
  - [Yi-VL-6B|34B](https://huggingface.co/collections/01-ai/yi-vl-663f557228538eae745769f3)
- Baichuan
  - [Baichuan-7|13B](https://huggingface.co/baichuan-inc)
  - [Baichuan2-7|13B](https://huggingface.co/baichuan-inc)
- BLOOM
  - [BLOOMZ&mT0](https://huggingface.co/bigscience/bloomz)
- Zhipu AI
  - [GLM-2|6|10|13|70B](https://huggingface.co/THUDM)
  - [CogVLM2-19B](https://huggingface.co/collections/THUDM/cogvlm2-6645f36a29948b67dc4eef75)
- OpenBMB
  - [MiniCPM-2B](https://huggingface.co/collections/openbmb/minicpm-2b-65d48bf958302b9fd25b698f)
  - [OmniLLM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
  - [VisCPM-10B](https://huggingface.co/openbmb/VisCPM-Chat)
  - [CPM-Bee-1|2|5|10B](https://huggingface.co/collections/openbmb/cpm-bee-65d491cc84fc93350d789361)
- RWKV Foundation
  - [RWKV-v4|5|6](https://huggingface.co/RWKV)
- ElutherAI
  - [Pythia-1|1.4|2.8|6.9|12B](https://github.com/EleutherAI/pythia)
- Stability AI
  - [StableLM-3B](https://huggingface.co/collections/stabilityai/stable-lm-650852cfd55dd4e15cdcb30a)
  - [StableLM-v2-1.6|12B](https://huggingface.co/collections/stabilityai/stable-lm-650852cfd55dd4e15cdcb30a)
  - [StableCode-3B](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650)
- BigCode
  - [StarCoder-1|3|7B](https://huggingface.co/collections/bigcode/%E2%AD%90-starcoder-64f9bd5740eb5daaeb81dbec)
  - [StarCoder2-3|7|15B](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a)
- DataBricks
  - [MPT-7B](https://www.databricks.com/blog/mpt-7b)
- Shanghai AI Laboratory
  - [InternLM2-1.8|7|20B](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
  - [InternLM-Math-7B|20B](https://huggingface.co/collections/internlm/internlm2-math-65b0ce88bf7d3327d0a5ad9f)
  - [InternLM-XComposer2-1.8|7B](https://huggingface.co/collections/internlm/internlm-xcomposer2-65b3706bf5d76208998e7477)
  - [InternVL-2|6|14|26](https://huggingface.co/collections/OpenGVLab/internvl-65b92d6be81c86166ca0dde4)

## SLM Data
- [LLMDatahub](https://github.com/Zjh-819/LLMDataHub) - a curated collection of datasets specifically designed for chatbot training, including links, size, language, usage, and a brief description of each dataset
- [Zyda_processing](https://github.com/Zyphra/Zyda_processing) - a dataset under a permissive license
comprising 1.3 trillion tokens, assembled by integrating several major respected
open-source datasets into a single, high-quality corpus

## SLM Evaluation:
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - A framework for few-shot evaluation of language models.
- [lighteval](https://github.com/huggingface/lighteval) - a lightweight LLM evaluation suite that Hugging Face has been using internally.
- [OLMO-eval](https://github.com/allenai/OLMo-Eval) - a repository for evaluating open language models.
- [instruct-eval](https://github.com/declare-lab/instruct-eval) - This repository contains code to quantitatively evaluate instruction-tuned models such as Alpaca and Flan-T5 on held-out tasks.
- [simple-evals](https://github.com/openai/simple-evals) - Eval tools by OpenAI.
- [Giskard](https://github.com/Giskard-AI/giskard) - Testing & evaluation library for LLM applications, in particular RAGs
- [LangSmith](https://www.langchain.com/langsmith) - a unified platform from LangChain framework for: evaluation, collaboration HITL (Human In The Loop), logging and monitoring LLM applications.  
- [Ragas](https://github.com/explodinggradients/ragas) - a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines.

## Miscellaneous
This repo contains awesome LLM paper list and frameworks for LLM training, tools to deploy LLM, courses and tutorials about LLM and all publicly available LLM checkpoints and APIs. Since SLM shares many of the same issues as LLM, I recommend that you also look at the contents related to LLM.
- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

## Contributing

This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for LLM, you could vote for them by adding 👍 to them.

---

If you have any question about this opinionated list, do not hesitate to contact me ro_keonwoo@korea.ac.kr.

[^1]: This is not legal advice. Please contact the original authors of the models for more information.