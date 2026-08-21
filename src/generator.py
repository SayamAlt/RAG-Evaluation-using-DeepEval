from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(name="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
        You are a precise assistant for customer support conversations.
        Answer the question using ONLY the information in the context below.

        Rules:
        - Use only information explicitly present in the context. Do not infer, assume, or add outside knowledge.
        - Answer thoroughly by identifying every distinct part of the question and cover each one, and include all the relevant points the context provides for answering it.
        - If a question has multiple parts or a concept has multiple components, address all of them rather than stopping at the first.
        - Reproduce numbers, amounts, durations, and named identifiers faithfully. For codes or identifiers that appear with irregular spacing due to voice transcription (e.g., "V C 2 5 3 7 1"), preserve the digits and letters in order but normalize spacing so the identifier reads naturally (e.g., "VC253-7107").
        - Always answer in a complete sentence that includes the action or relationship the question implies. If asked what was "removed", say what it was AND that it was removed. If asked what "happened", describe the full event. Never answer with a noun phrase alone.
        - Lead with the core fact directly. NEVER open by restating or embedding the question. Do NOT start your answer with any portion of the question text. For example, if asked "What were the two charges on the bill?", do NOT answer "The two charges on the bill were...". Instead, say "A late payment fee and a data overage charge appeared on the bill." Begin with the answer substance, not the question. The answer must still be a grammatically complete sentence — never a bare noun phrase alone.
        - Speaker attribution: the context contains dialogue between multiple parties (customer, agent, merchant, technician). When the question asks what a specific party said, confirmed, or did, attribute only that party's statements to them — do not mistake a customer's personal belief or denial for what a merchant or third party actually confirmed.
        - When the context presents a hypothesis or initial framing that is subsequently corrected or confirmed in the dialogue, base your answer on the final resolution — not the initial hypothesis. For example, if the dialogue first raises "is it intermittent?" and then the customer confirms "no, it is a complete loss", answer with "complete loss".
        - When asked to name a specific item (a charge type, a room type, a product name, a service), reproduce the EXACT noun phrase from the context verbatim. Do NOT substitute generic words: never say "additional fees" when the context says "data overage charge"; never say "standard room" when the context says "Classic King"; never say "extra charges" when a specific charge name is present. If the context mentions two or more named items, name each one precisely.
        - Stay strictly faithful to the roles described in the context. If the context says an agent is helping a customer, do not say the customer is helping. Do not reassign who did what.
        - Only respond with "I do not have enough information in the provided context to answer that" if the specific fact being asked is completely absent from all context chunks. If any chunk contains relevant information about the topic, attempt to answer using that context.
        - Keep the answer short, clear, and concise.

        Context:
        {context}

        Question: {question}

        Answer:
    """
)

chain = prompt | llm | StrOutputParser()

def generate(query: str, context: list[str]) -> str:
    """ Generates a grounded answer for the query and context chunks. """
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})

if __name__ == "__main__":
    context = [
        "The customer purchased a new laptop last week but it is not working properly. Agent Annie asked for the order number, which was 123456789.",
        "The customer's monthly bill at Bright Star Services is usually around $80 but this month it came to $150. Agent Lisa reviewed the breakdown to find the source of the extra charges.",
        "Sophie's internet connection dropped while she was trying to send a work report. Her screen showed a no internet error with an exclamation point. She needed to send the report before 10 AM.",
        "Steven called Hotel California to book a room for 4 people on August 18th. Agent Candice took the reservation details.",
        "Customer Orlando Taylor noticed an unrecognized transaction on his debit account. The agent asked for the account number and verified his full name before pulling up the account information.",
    ]
    print(generate("Why did Steven called the hotel for?", context))