
Aspect-Based Sentiment Analysis is a classic problem in natural language processing. From the perspective of sentiment objects, its granularity is primarily reflected in the sentiment polarity classification of
each aspect within a sentence. The essence of this task lies in identifying the sentiment objects in a sentence
and determining their corresponding semantically relevant context. There has been a significant body of
research conducted on Aspect-Based Sentiment Analysis, encompassing diverse approaches ranging from
early machine learning single-model techniques to advanced deep learning models. Additionally, the field
has witnessed the evolution from manual dictionary mapping to the utilization of highly versatile pre-trained
models. Each model has its own strengths and limitations. This paper aims to delve into Aspect-Based Sentiment Analysis by exploring and proposing several distinct approaches. Furthermore, improvement ideas
and methods will be presented. The main contents of this paper include the following:
1. We focus on efficient exploration and practical implementation of early methods for Aspect Based
Sentiment Analysis. The task is divided into two subtasks: aspect term extraction and aspect term
classification. Machine learning models, specifically random vector fields and support vector machines, are employed for training and prediction. The performance of these models is compared and
discussed in comparison to other classical models.
2. in single model machine learning, deep learning methods such as convolutional neural networks are
employed for processing. Gated units are utilized to filter and identify aspect terms, integrating the
two subtasks into a unified network and achieving superior classification performance.
3. Considering the problem of convolutional neural networks relying heavily on sentence structure for
context extraction, graph convolutional neural networks with gated units used for signal selection are
proposed. They are combined for feature abstraction and classification. The extraction and representation methods of semantic dependency graphs are introduced. Finally, the sentiment classification
results are compared with convolutional neural networks, demonstrating the strong abstraction ability of the structural semantic dependency graph in text classification. Besides, we discuss how to
choose the gate units and explain the choices
