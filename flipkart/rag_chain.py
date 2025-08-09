from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

# This file creates a shopping chatbot that remembers past conversations.
# It finds relevant product reviews and uses them to answer questions clearly.

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL , temperature=0.5)
        self.history_store={} #will sotre all conversations for all sessions/chats

    #it is a private method (underscore before method name)
    #this method is to get all the chat history for a particular session
    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):

        #here k is relevant documents to search (each row is a separate doc)
        #supppose you ask suggest me gaming headphones, and you have 50 headphones in total
        #so retriever will firstly fetch gaming headphones from all headphones and it will search for 3 gaming headphones bcz k=3
        #so finally we will get k gaming headphones
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        #Suppose, uh, you are the user and you, uh, you have a chatbot.
        #Suppose user ask for something like, uh, how is the camera on iphone14 phone?
        #Bot answers: The camera is very good. (No context prompt applicable till now)
        #User again asks: what about battery life?
        #Bot will not immedetialy answer, it will resdesign the question for hmself only
        #what about battery life on iphone14 (context is used) and then it will answer to user
        #bot: Battery life is decent
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])


        #This is normal promt that we are telling the bot 
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""), #this is changed input from context
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  #This input is given by user
        ])

        # Creates a retriever that can rewrite user queries using chat history
        # Example:
        # User: "How is the camera on iPhone 14?"
        # Bot: "The camera is very good."
        # User: "What about the battery?"
        # → This retriever will turn the last question into:
        #   "What about the battery on iPhone 14?" (using context_prompt above)
        #Earlier we formed prompt, now we are actually using it
        history_aware_retriever = create_history_aware_retriever(
            self.model , retriever , context_prompt
        )

        # Creates a chain that takes retrieved documents (reviews) + qa_prompt
        # and sends them to the model for answering the question.
        # Essentially: "Take the top-k relevant reviews, format them into the
        # prompt, and let the AI answer."
        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )

        # Combines the history-aware retriever + question-answer chain
        # This means:
        #   1. Rewrite question using history.
        #   2. Retrieve relevant docs from vector store.
        #   3. Pass them to the QA chain for answer generation.
        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        # Wraps the whole chain so that:
        #   - It stores and retrieves past conversation history
        #   - Each session_id gets its own independent history
        #   - History is automatically fed into retriever and QA chain
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input", # User's question key
            history_messages_key="chat_history", # Past conversation key
            output_messages_key="answer"    # Bot's reply key
        )


