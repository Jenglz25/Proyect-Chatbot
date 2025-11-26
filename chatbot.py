# chatbot.py
class Chatbot:
    def __init__(self, retriever, llm_engine, df, top_k=5):
        self.retriever = retriever
        self.llm = llm_engine
        self.df = df
        self.top_k = top_k

        # Crear resumen automático del dataset
        self.dataset_description = self.build_dataset_description()

    def build_dataset_description(self):
        desc = "DESCRIPCIÓN DEL DATASET:\n"
        desc += f"- Número de filas: {len(self.df)}\n"
        desc += f"- Número de columnas: {len(self.df.columns)}\n\n"
        desc += "COLUMNAS DISPONIBLES:\n"

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            desc += f" • {col} (tipo: {dtype})\n"

        return desc

    def ask(self, user_query):
        """
        Reglas:
        1. Si la pregunta es sobre el dataset → NO uses retrieval.
        2. Si la pregunta es sobre contenido de trabajos → SÍ usamos retrieval.
        3. Siempre envía el dataset_description en el prompt.
        """

        # ¿El usuario pregunta sobre el dataset?
        dataset_keywords = ["dataset", "archivo", "csv", "columnas", "estructura",
                            "qué datos", "qué información", "qué contiene"]

        is_dataset_question = any(kw in user_query.lower() for kw in dataset_keywords)

        if is_dataset_question:
            context = self.dataset_description
            retrieved_docs = []
        else:
            retrieved_docs = self.retriever.search(user_query, k=self.top_k)
            context = self.dataset_description + "\n\n" + "\n\n".join(retrieved_docs)

        # LLM responde usando el contexto
        response = self.llm.generate(context, user_query)
        return response, retrieved_docs
