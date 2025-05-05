# Taiwan CivilCode Agent

此專案目標為嘗試在 AgentFlow 中應用 CoT 來增進小模型的法規問題推理能力。

Jupyter Notebook Version [1223-MultiHop-RAG](https://github.com/hank1224/DataTeam-RAG-training/tree/main/1223-MultiHop-RAG).

Preview Workflow: [LangSmith Log](https://smith.langchain.com/public/801b3911-1617-41c3-86d4-050e740c4732/r)

![LangGraph_Studio_Preview](./static/LangGraph_Studio_Preview_0430.png)

## TODO

1. Use [LangGraph multi-agent Network](https://langchain-ai.github.io/langgraph/how-tos/#multi-agent) to replace `ThreadPoolExecutor` in [nodes.py: execute_parallel()](src/taiwan_civilcode_agent/nodes.py)

2. Build the Knowledge Graph of CivilCode for searching.

3. 把解題關鍵納入考慮，判例、法條修改歷史...等，也就是領域知識，進而修改架構。

### Multi-Agent Monte Carlo Voting | Best of N
辨別選擇題類別，根據類型採取不同的解題策略

【策略 A：直接信息檢索與驗證】 (合併原 1 & 2)
- 適用題型：1 (定義/概念識別), 2 (數值/事實記憶)。
- 核心任務：從知識庫（法條、記憶）中查找特定的、單一的、固定的信息點（定義、數字、日期、名稱等）。
- 執行流程：識別關鍵詞 -> 查詢資料 -> 匹配選項。

【策略 B：規則應用與演繹推理】 (保留原 3)
- 適用題型：3 (規則應用/情境判斷)。
- 核心任務：理解具體情境，識別適用規則，進行邏輯推導，判斷法律效果或行為定性。
- MC 應用：多 Agent 獨立進行情境分析和規則推演，對可能的結果進行投票，找出最符合邏輯和規則的結論，處理複雜或模糊情況。
- 執行流程：解析情境 -> 定位規則 -> 多 Agent 並行推演 -> 投票/綜合判斷 -> 匹配選項。

【策略 C：多選項屬性評估與篩選】 (合併原 4, 5, 6)
- 適用題型：4 (因果/關係判斷), 5 (多重敘述真偽判斷), 6 (排除法尋找例外)。
- 核心任務：逐一評估每個選項是否具備題目所關注的特定屬性（如：真實性、與某概念的關係、是否屬於某類別）。
- MC 應用：針對每個選項，多 Agent 獨立判斷其是否滿足該屬性（True/False, Related/Unrelated, Belongs/Doesn't Belong, Correct Relationship/Incorrect Relationship）。投票決定每個選項的最終評估結果。
- 執行流程：
    1. 明確評估屬性（由題型決定：真偽？關係？歸屬？）。
    2. For each option: 多 Agent 並行評估該選項是否符合屬性 -> 投票確定該選項的屬性狀態。
    3. 根據題目要求（尋找正確的、錯誤的、不屬於的、描述了正確關係的等），從評估結果中篩選出最終答案。

【策略 D：比較權衡與最優選擇】 (保留原 7)
- 適用題型：7 (比較/最適判斷)。
- 核心任務：在多個選項間進行比較，根據特定標準（最有效、最優先、最符合、程度最高等）選出最佳選項。
- MC 應用：多 Agent 獨立根據標準對各選項進行評分、排序或比較判斷，綜合結果（如平均分、多數票）確定哪個選項最優。
- 執行流程：確定比較標準 -> 多 Agent 並行評估/比較所有選項 -> 綜合評估結果 -> 選出最優選項。

### 專案名稱建議：

1.  **Adaptive Multi-Agent Voting Solver (AMAVS)**
2.  **Stratified Consensus System for Multiple Choice (SCS-MC)**
3.  **Intelligent MCQ Strategy & Voting Engine**
4.  **Best-of-N MCQ Resolver with Dynamic Strategy**
5.  **Multi-Agent Stratified MCQ Solver**

### 專案描述：

這是一個**基於多代理協作投票機制 (Multi-Agent Voting)** 的智能選擇題解決系統。系統的核心在於根據選擇題的具體類型（如：事實記憶、規則應用、多項判斷、最優選擇等），**動態地選擇並執行最適合的解題策略**。

在需要複雜推理、多角度評估或比較判斷的題型（對應策略 B, C, D）中，系統會啟動**多個獨立的 AI 代理 (Agents)**。這些代理會並行地對題目情境、選項屬性或比較標準進行獨立分析和判斷。最終，系統會採用**投票或共識機制 (Voting / Consensus)** 來綜合這些代理的獨立判斷結果，得出最可靠或最優的答案。

對於直接信息檢索類的題型（策略 A），系統則採取更直接的資料查詢與驗證方式。

透過這種**策略分層 (Stratified Strategy)** 與**多代理協作 (Multi-Agent Collaboration)** 相結合的方式，旨在提升選擇題解答的準確性、魯棒性（抗干擾能力）以及處理複雜或模糊問題的能力。

### 核心價值 (Core Value)：

此專案的核心價值在於**提升選擇題解答的「準確性」與「魯棒性」**，特別是對於那些需要複雜推理、多維度分析或存在一定不確定性的題目。

具體來說，它實現了以下價值：

1.  **智能化策略適應 (Intelligent Strategy Adaptation):** 系統不再使用單一僵化的方法，而是根據題型特徵切換最適策略，提高了效率和針對性。
2.  **增強的決策可靠性 (Enhanced Decision Reliability):** 通過多個獨立代理的並行分析與投票機制，減少了單一模型或單一推理路徑可能帶來的偏誤和錯誤，使最終決策更接近「集體智慧」的結果，增加了答案的可靠性。
3.  **處理複雜性與模糊性 (Handling Complexity & Ambiguity):** 多代理協作特別適用於需要從多個角度考量或判斷存在一定模糊性的情況，提高了系統處理困難題型的能力。
4.  **提高系統的抗干擾能力 (Improved Robustness):** 即使個別代理出現判斷偏差，投票機制也能有效地篩選掉這些異常，使系統整體表現更穩定。

總之，這個專案的核心價值是利用**「策略的智慧分層」**與**「多代理的協同增強」**來構建一個更為**精準、可靠且適應性強**的自動化選擇題解答系統。


## Data

民法全文（110-01-20版本），條文編號：1~1125，總計共1439筆。[全國法規資料庫 - 民法](https://law.moj.gov.tw/LawClass/LawAllPara.aspx?pcode=B0000001)

資料清洗過程詳見：[資料清洗.md](https://github.com/hank1224/DataTeam-RAG-training/tree/main/1223-MultiHop-RAG/data-pre-process)

## Evaluation

使用國考選擇題考題，選用 `113年公務人員特種考試司法人員、法務部調查局調查人員及海岸巡防人員考試` 之選擇題部分。

考題需與當時的民法版本相符，此問題已被考慮並且已選用適合的版本。

- 民法：110-01-20 修訂版本
- 試卷：113年