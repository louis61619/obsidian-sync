# langchain-RAG

  

在構建 AI 應用時，比如對話型企業知識庫、網站導航助手等等，通常都會需要對大模型所能夠獲取到的資料進行擴展，接下來會對如何做到這件事進行分析。

  

### 如何擴展大模型的資料庫

  

1. 將資料都塞進上下文中

2. 使用 RAG (檢索增強生成)

3. 訓練自有模型或對現成的開源模型進行微調 [fine tuning](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/fine-tuning-overview)

  

隨著大模型的進步所能傳過去的資料也越來越多，不過基本上大量的 token 意味著價格提高，也仍舊有著內容的上限，而訓練自有模型幾乎不會考慮、微調模型要花費的時間和成本太高，選擇 RAG 並結合提示詞工程是綜合考量上不錯的選擇

  

### 建立 RAG

  

RAG 知識庫建立三步

chunk：數據切片，在有大量資料的場景，需要將資料進行分割

Embedding：向量化，為了之後使用算法進行比對檢索

Vector Database：向量資料庫，用於存儲向量化的資料

  

基本上以上這三步都用開源庫去完成就好

  

RAG 檢索三步

Retrieval：檢索向量資料庫，透過算法檢索到相似數據返回（這一步會將用戶輸入的內容 Embedding 然後透過算法找到對應的資料庫中的某些結果）

Augmented：將找到的資料和用戶輸入的問題組成一個大模型（chatgpt）能理解的 prompt

Generation：輸入給大模型並得到回答

  

基本 RAG

  

![naive rag](assets/image.png)

  

為了更加精準的找出使用者真正需要的回答並擴展使用情境，可以加入

[query_transformations](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations/)：可以透過一些工具將使用者的問題精簡成幾段準確的字句，將問題切割成不同小問題

query routing：查詢路由，使用者所需要的數據可能並非只會從向量料庫中獲取，可以從其他資料庫比如 mysql、甚至爬取其她網站或文檔中讀取，這時候將使用者的問題進行分類，後續可以由專屬於這類問題的流程進行處理

  

進階 RAG

  

![advanced rag](assets/advance-rag.png)

  

還有些常見的方式可以增加 RAG 的準確性和效率，比如使用假設性問題去匹配使用者的問題，先建好預設的回答再透過llm去匹配

  

### 實作

  

講一下 pdf 搜索，查找，回答

講一下爬蟲搜索，查找，回答，範例網站：https://www.platformatichq.com/node-principles

講一下實現一個簡單的問答的bot

  

https://www.nature.com/nature/volumes


切割
```js
import { CharacterTextSplitter } from 'langchain/text_splitter';

const text = `
Hi.
I'm KKKKKKKK
How? Are? you?
Okay then dfiffi.
122223334444
what do you do?
what is the wheather?
`;

const splitter = new CharacterTextSplitter({
  separator: '\n', // 分割符
  chunkSize: 20, // 分割的塊的size
  chunkOverlap: 6, // 分割的塊增加一些信息的冗餘
});

const output = await splitter.createDocuments([text]);

console.log(output);
```

向量化
```js
import { embeddingModel } from '../../utils/utils.mjs';

const documentRes = await embeddings.embedDocuments(["Hello world", "Bye bye"]);

console.log(documentRes);
```

儲存至向量資料庫
```js
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { embeddingModel } from '../../utils/utils.mjs';

const vectorstore = await MemoryVectorStore.fromTexts(
  ['Hello world', 'Bye Bye', 'nice world'],
  [{ id: 2 }, { id: 1 }, { id: 3 }],
  embeddingModel,
);

const res = await vectorstore.similaritySearch('hello', 1);
console.log(res);
```

使用向量資料庫實現RAG
https://js.langchain.com/docs/integrations/vectorstores/hanavector/#using-a-vectorstore-as-a-retriever-in-chains-for-retrieval-augmented-generation-rag
```js
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { chatModel } from '../../utils/utils.mjs';
import { loadedVectorStore } from './close-vector-load.mjs';

const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are an expert in state of the union topics. You are provided multiple context items that are related to the prompt you have to answer. Use the following pieces of context to answer the question at the end.\n\n{context}',
  ],
  ['human', '{input}'],
]);

const combineDocsChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt: questionAnsweringPrompt,
});


const chain = await createRetrievalChain({
  retriever: loadedVectorStore.asRetriever(),
  combineDocsChain,
});

  

const response = await chain.invoke({
  input: 'Who is the article about?',
});

  
console.log('Chain response:');

console.log(response.answer);
```

基於 query routing 實現多情境處理
https://js.langchain.com/docs/how_to/routing/

  

### 參考資料

  

llm工具庫

https://github.com/langchain-ai/langchainhttps://github.com/run-llama/llama_index

  

提示詞工程指南

https://platform.openai.com/docs/guides/prompt-engineering

  

RAG

https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6https://juejin.cn/post/7409191765708947465?searchId=202410010941272077808BF0A070976412https://www.falkordb.com/blog/advanced-rag/

  

coze

https://juejin.cn/post/7355026320088121381?searchId=20241001160139C40AB462929CA9AFC206