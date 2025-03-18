# Vietnamese Machine Reading Comprehension

**Abstract:** Extraction-based machine reading comprehension tasks typically generate answers as single spans from a passage. However, real-world answers often involve multiple spans from different positions. Relying on single-span answers can omit crucial information, include irrelevant details, or cause grammatical errors. Multi-span answers can address these issues, but datasets for multi-span tasks remain limited. In this study, we constructed a comprehensive multi-span reading comprehension dataset, consisting of 1,457 question-answer pairs. Using BERT, our best results achieved 43.85% accuracy, 58.59% ROUGE-L, and 82.06% BERTScore-F1. We also analyzed error cases to guide future improvements.<br>

## Introduction
<p align="center">
<img width="450" alt="{9D0CC30A-FC44-4F3C-B784-6D76717F1BBD}" src="https://github.com/user-attachments/assets/0b36b4a4-0149-4ab0-a75d-8ca9cb6c6f27">
</p>

**INPUT**:

**Passage**: Người gốc Mỹ Latinh hoặc Iberia *là nhóm cư dân*
tại Texas có số dân lớn thứ hai sau người gốc Âu không có
nguồn gốc Mỹ Latinh và Iberia. Có trên 8,5 triệu người tuyên
bố rằng mình thuộc nhóm dân cư này, chiếm 36% dân cư Texas.
Trong đó, 7,3 triệu *người có nguồn gốc México*, chiếm 30,7% cư
dân. Có trên 104.000 người *Puerto Rico* và gần 38.000 người
Cuba sinh sống trong bang. Có trên 1,1 triệu người (4,7% cư
dân) có tổ tiên Mỹ Latinh hoặc Iberia khác nhau, như người
Costa Rica, Venezuela, và Argentina. ( Latinx or Iberian-origin
individuals in Texas *are an ethnic group* that ranks second in
size, following non-Latinx individuals of European descent.. Over
8.5 million people identify themselves as part of this population,
comprising 36% of Texas’ population. Among them, 7.3 million
have *Mexican ancestry*, making up 30.7% of the residents. There
are over 104,000 individuals from *Puerto Rico* and nearly 38,000
from Cuba residing in the state. Additionally, over 1.1 million
people (4.7% of the population) have diverse Latinx or Iberian
ancestry, including individuals from countries like *Costa Rica,
Venezuela, and Argentina*.)

**Question**: Những người *gốc Mỹ Latinh sống ở Texas* những
quốc gia nào? (Which countries do individuals *of Latinx origin
living in Texas* come from?)

**OUTPUT**:

**Answer**: người có nguồn gốc México, Puerto Rico, Cuba, Costa
Rica, Venezuela, và Argentina là nhóm cư dân gốc Mỹ Latinh
sống ở Texas (Mexican ancestry, Puerto Rico, Cuba, Costa Rica,
Venezuela, and Argentina are an ethnic group of Latinx origin
living in Texas)

## Dataset Statistics
We aim to build and contribute to the "Multi-Span Question
Answering in the Vietnamese Language" dataset to promote
the development of natural language processing models for
the Vietnamese language and expand their applications in
other areas such as chatbots, information extraction, and
text summarization in Vietnamese.

After completing dataset construction, we obtained 1457
question-answer pairs, divided into the training, validation,
and test sets in an 8:1:1 ratio. Our dataset consists of 1457 question-answer pairs. Each
answer corresponds to the provided question of the passage.
The domain of our dataset spans various subjects, including
history, geography, science, etc.

<p align="center">
<img width="440" alt="{FB49C4E4-D19C-4A76-A9DA-AF5A4DACB0BD}" src="https://github.com/user-attachments/assets/79f04168-6c11-4ee4-88ad-e6e0ab33f4ad">
</p>


| Question Type | Train | Validation | Test | Full |
|---------------|-------|------------|------|------|
| How           | 108   | 9          | 12   | 129  |
| What          | 379   | 48         | 44   | 471  |
| Why           | 90    | 10         | 14   | 114  |
| Where         | 55    | 7          | 6    | 68   |
| When          | 67    | 7          | 12   | 86   |
| Which         | 256   | 38         | 32   | 326  |
| Who           | 105   | 14         | 6    | 125  |
| Other         | 105   | 13         | 20   | 138  |



|   Reasoning Type       | Train | Validation | Test | Full |
|------------------------|-------|------------|------|------|
| Word matching          | 301   | 38         | 38   | 377  |
| Paraphrasing           | 425   | 53         | 53   | 531  |
| Math                   | 69    | 9          | 8    | 86   |
| Logic/causal relation   | 173   | 21         | 22   | 216  |
| Coreference            | 197   | 25         | 25   | 247  |


| Maximum Length          | Train | Validation | Test |
|---------------------|-------|------------|------|
| Passage             | 474   | 374        | 386  |
| Question            | 40    | 43         | 32   |
| Passage + Question  | 490   | 381        | 406  |
| Answer              | 81    | 92         | 63   |

| Average Length        | Train  | Validation | Test   |
|-----------------------|--------|------------|--------|
| Word Matching         | 22.41  | 19.95      | 20.97  |
| Paraphrasing          | 21.58  | 22.01      | 19.19  |
| Math                  | 16.72  | 15.33      | 15.75  |
| Logic/Causal Relation  | 25.47  | 27.1       | 26.45  |
| Coreference           | 22.45  | 24.48      | 18.2   |


## Results

We conducted experiments using the BERT-base-
multilingual-cased model with many of the hyperparameter
set. The overall model performance on the validation
set achieved 45.22% with BLUE1. The ROUGE-L and
BERTScore-F1 scores reached 59.13% and 82.12%, respectively. The model achieved 43.85% for the test set, with
58.59% for ROUGE-L and 82.06% for BERTScore-F1.

Additionally, we evaluated the number of spans of answer
aspects in the validation and test sets. In both datasets,
answers containing only one span performed significantly
worse than answers with multiple spans. Specifically, in the validation set, single-span answers scored 16% lower in
BLUE1, 29.52% lower in ROUGE-L, and 7.01% lower in
BERTScore compared to multi-span answers. In the test
set, multi-span answers outperformed single-span answers
by 8.83% in BLUE1, 11.68% in ROUGE-L, and 2.23% in
BERTScore.

The lower performance of single-span answers in both
datasets can be attributed to the dominance of multi-
span answers in the training set. As a result, the model
has learned and tends to predict answers with multiple
spans and more words. On the other hand, single-span
answers have fewer words, leading to lower BLEU1 and
ROUGE-L scores compared to multi-span answers in both
the validation and test sets. Regarding BERTScore, the
predicted answers contain more words, resulting in different
contexts than single-span answers in the two datasets.
Consequently, the performance of single-span answers is
lower than that of multi-span answers.
<p align="center">
<img width="800" alt="{E8CDBFCB-57DB-44C1-9B79-30A3F3993BB5}" src="https://github.com/user-attachments/assets/0474d6d6-541c-4beb-bb8b-7fe31e62afba">
</p>

<p align="center">
<img width="800" alt="{421B9B03-87FA-4762-8427-B54DEF1C3ECA}" src="https://github.com/user-attachments/assets/0c00d6d1-9529-4c65-9127-cce697bdc441">
</p>

<p align="center">
<img width="400" alt="{52B8CF90-6C3A-4DB9-B34C-AE38A9EB518F}" src="https://github.com/user-attachments/assets/8254206d-6801-42ba-bba2-c34fe4a591b5">
</p>





