# used in FAQs identification
representative_question_prompt = """
    The following documents are a subset of representative questions that are part of the same topic: [DOCUMENTS]
    The topic is described by the following representative keywords: [KEYWORDS]

    Based on the above information, can you extract a concise question that best captures the essence of the topic?
    The question should be no longer than 10 words and it should follow the format: <Representative question?>
    """
