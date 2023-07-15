# "Insights into ChatGPT Research: A Comprehensive Analysis"

The objective of this project was to conduct a comprehensive analysis of research papers related to ChatGPT by scraping data from Google Scholar. The dataset obtained consisted of numerous research papers covering various aspects of ChatGPT, and the analysis aimed to gain insights into the research landscape surrounding this topic.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/ea0c2119-53ec-4a49-a47e-49c533e8276b" alt="image"  width="800" height="400">


___

# Table of Contents

- [Background](#background)
- [Objective](#objective)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Data Modeling](#data-modeling)
- [Results](#results)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Topic Modeling](#topic-modeling)
- [Conclusion](#conclusion)
- [References](#references)

___

# Background

ChatGPT has emerged as a groundbreaking technology in the field of natural language processing (NLP) and conversational AI, revolutionizing the way humans interact with computers and transforming various domains. With its advanced language generation capabilities, ChatGPT has shown tremendous potential for enhancing human-computer interactions and enabling more intelligent and intuitive conversational experiences.
In recent years, there has been a significant surge in the development and adoption of conversational AI systems, driven by the increasing demand for personalized and natural language-based interfaces. ChatGPT, developed by OpenAI, represents a major milestone in this domain, showcasing the power of deep learning and neural networks in generating human-like responses and engaging in meaningful conversations.
The importance of ChatGPT lies in its ability to comprehend and generate human language with remarkable accuracy and coherence. Through the utilization of large-scale pre-training on vast amounts of text data, ChatGPT has been trained to understand the nuances of language, context, and semantics, allowing it to generate responses that closely resemble those of a human conversational partner.
The applications of ChatGPT are diverse and far-reaching. In customer service, it has the potential to enhance the quality of interactions by providing intelligent and efficient responses to user queries. In educational settings, ChatGPT can serve as a virtual tutor, providing personalized assistance and guidance to students. Furthermore, ChatGPT holds promise in fields such as healthcare, information retrieval, and content generation, where it can augment human capabilities and streamline processes.
The significance of ChatGPT extends beyond its immediate applications. It represents a significant advancement in the broader field of NLP, pushing the boundaries of language understanding and generation. The challenges associated with building a conversational AI system that can comprehend and respond effectively to human inputs are immense, requiring sophisticated algorithms, vast amounts of data, and ongoing research efforts. As such, ChatGPT serves as a benchmark for evaluating the progress and capabilities of NLP models and techniques.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/be30b99b-a982-4bde-9695-76e51d60f6f4" alt="image" width="400" height="300">

___

# Objective

The focus of this study is the research landscape surrounding ChatGPT, with the objective to deliver a thorough analysis of predominant themes, research trends, and potential areas of exploration within this domain. This project involves extensive data collection, exploratory data analysis, and topic modeling to identify primary areas of interest, recognize unexplored research avenues, and generate insights to steer future studies.

This research aims to contribute to the growing body of knowledge surrounding ChatGPT, providing researchers, practitioners, and decision-makers with valuable insights into the advancements, challenges, and opportunities within the realm of conversational AI. By understanding the current state of research, we can further harness the potential of ChatGPT, pushing the boundaries of human-computer interactions and paving the way for more intelligent and intuitive AI systems.

___

# Methodology 

## Data Collection

The initial step involved data collection, wherein I leveraged ScraperAPI to scrape Google Scholar and amass a dataset of approximately 1000 research papers related to "ChatGPT". ScraperAPI proved instrumental in handling proxies, browsers, and CAPTCHAs, enabling me to retrieve the HTML content of web pages through a straightforward API call. The search query was specifically tailored to "ChatGPT", and I extracted various data points including paper titles, links, paper snippets, citation details, and publication information from the first 100 pages of Google Scholar on March 29, 2023. To circumvent potential breakdowns or IP blocks, I executed the code iteratively for a total of 10 times, scraping only 10 pages per iteration. Ultimately, I concatenated all the extracted data from the CSV files into a single dataset.

## Data Preprocessing

In the preprocessing phase, I employed various techniques to enhance the quality of the data. Firstly, I utilized the detect function from the langdetect library to filter out non-English titles, ensuring that the analysis focused solely on English-language papers. Next, I employed regular expressions (regex) to eliminate occurrences of [HTML][HTML] or [PDF][PDF] at the beginning of paper titles, as they were artifacts of the scraping process. To further refine the data, I leveraged several components from the Natural Language Toolkit (NLTK) library, including the removal of punctuation and stop words using word_tokenize and stopwords functions, respectively. 
Additionally, I eliminated the frequently occurring term "ChatGPT" from the titles, as it represented the overarching topic of my analysis. Furthermore, I applied lemmatization and stemming techniques using the WordNetLemmatizer and SnowballStemmer functions to normalize the remaining words, reducing them to their base form. These preprocessing steps ensured that the dataset contained relevant and standardized textual information for subsequent analysis. In addition to the paper titles, I also applied the same preprocessing steps to the paper snippets.

### EDA

For the exploratory data analysis (EDA), I performed several key steps to gain insights from the cleaned text data. First, I combined all the cleaned titles and snippets into a single list. This allowed me to work with a unified dataset for further analysis.

Next, I utilized the Counter module from the collections library to count the frequency of each word in the cleaned text. By examining the word frequencies, I was able to identify the most common terms and gain a general understanding of the content.
To visualize the top 20 words, I employed the LetsPlot library, using the setup_html() function to generate an interactive bar chart. This visualization provided a clear representation of the most frequent words, enabling me to identify prominent themes or topics within the ChatGPT research domain.

Furthermore, I utilized the WordCloud module from the wordcloud library to create a visually appealing word cloud based on the word counts. This allowed me to observe the relative prominence of different words in a visually striking manner.

To gain insights into the citation counts of the papers, I utilized the search function from the regex library to extract the citation count information from the scraped data. I then employed the sns module from the seaborn library to visualize the top 10 cited papers, providing a glimpse into the most influential works in the field.

moreover, I employed the str.extract function to extract publication information such as the publication year and the names of the research networks or science databases on which the papers were published. This information allowed me to analyze the distribution of publications across different platforms and track the temporal trends of ChatGPT research.

lastly, To gain insights into the sentiment expressed in the text data, I utilized the TextBlob module from the textblob library for sentiment analysis. This allowed me to assess the polarity and subjectivity of the paper titles and snippets, providing an understanding of the overall sentiment associated with the ChatGPT research. 

Performing these EDA techniques led to gaining valuable insights into the cleaned text data, identifying significant words and topics, exploring citation patterns, and examining publication information. These findings laid the foundation for further analysis and interpretation of the ChatGPT research landscape.


## Data Modeling

For the topic modeling phase, I employed several libraries to perform LDA (Latent Dirichlet Allocation) topic modeling on the collected data. The libraries used included gensim, pyLDAvis.gensim_models, and corpora from gensim.
To begin, I created a dictionary and a Bag-of-Words (BoW) corpus using the gensim library. These representations allowed me to transform the textual data into a numerical format suitable for topic modeling analysis.
To determine the optimal number of topics, I followed an iterative approach. I built multiple LDA models with different values of the number of topics (k) and assessed the coherence value for each model. The coherence value serves as an indicator of the interpretability and meaningfulness of the topics generated. Typically, selecting a value of k where the coherence score reaches its peak and exhibits a diminishing growth pattern leads to the most meaningful topics. However, if the same keywords are repeated across multiple topics, it suggests that k may be too large.
To automate this process, I developed a function that takes in the Gensim dictionary, Gensim corpus, input texts, and a range of potential topics. This function returns a list of models and their corresponding coherence values, aiding in the selection of the optimal number of topics.
In this case, I created ten LDA models with different numbers of topics and evaluated the coherence scores for each model. Visualizing the coherence scores for different numbers of topics (ranging from 2 to 10), I observed that five topics yielded the highest coherence score of 0.593.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/7fdc27ec-c1d3-409b-ba6b-1e3fab8de588" alt="image" width="500" height="400">

Furthermore, I focused on refining the hyperparameters to improve the model's performance. I specifically looked at the hyperparameters alpha and beta, which respectively control the document-topic density and word-topic density. After experimentation, I found that setting both alpha and beta to symmetric values resulted in the best performance, with a coherence score improvement of about 2% over the baseline model.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/6335c6b7-bc82-42c9-8424-26b283b16460" alt="image" width="500" height="400">

With the optimal hyperparameters determined, I created the final LDA model with a coherence score of 0.593. To visualize the topics and their distribution, I utilized pyLDAvis.gensim_models and the LDAvis_prepared function. This allowed me to generate an interactive visualization of the topics and their corresponding keywords, facilitating a better understanding of the relationships and characteristics of each topic.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/2fc7b1d5-741a-41bd-8f37-2b2ef8807f3d" alt="image" width="500" height="400">

By leveraging these techniques and libraries, I successfully performed LDA topic modeling on the ChatGPT research papers, ultimately obtaining a coherent and interpretable set of topics that encapsulated the main areas of focus within the dataset.

___

# Results

## Exploratory Data Analysis

### 1.Word-Frequency Analysis

The word-frequency analysis conducted on nearly 1000 Google Scholar papers focused on ChatGPT revealed some intriguing insights. The result of the analysis is visually represented in a bar graph, which showcases the most frequently used words and their corresponding frequencies, allowing for a quick and intuitive understanding of the research trends and focus areas within the field. Keywords such as “use” (209 occurrences, 1.48%) and “model” (201 occurrences, 1.43%) stand out prominently, additionally the word “language” appeared 157 times (1.19%) together emphasizing the practical application and implementation of ChatGPT as a language model. While “write” (161 occurrences, 1.14%), “artificial” (147 occurrences, 1.04%), and “intelligence” (133 occurrences, 0.94%) highlight the focus on generating written content and exploring artificial intelligence aspects. The visualization also depicts the frequent usage of terms like "research," "study," and "education" (each occurring 104 times, 0.74%), illustrating the growing interest in utilizing ChatGPT within educational and research settings. Furthermore, it illustrates the importance of ChatGPT as a tool (100 occurrences, 0.71%) and its potential to serve as a conversational agent (82 occurrences, 0.58%). The visualization effectively communicates these findings, providing a concise and accessible summary of the word-frequency analysis results.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/c1fccd56-2005-47fa-96c2-c8f29fc3445d" alt="image" width="500" height="400">

### 2.Citation Analysis

The visualization of the citation analysis reveals the top cited papers in the field of ChatGPT research. Among them, the paper titled "Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models" by TH Kung et al. stands out with a citation count of 87. It discusses the evaluation of ChatGPT's performance on the United States Medical Licensing Examination (USMLE) and highlights its ability to achieve passing or near-passing results. Another highly cited paper is "ChatGPT is fun, but not an author" by HH Thorp, with a citation count of 80. It explores the use of ChatGPT in writing scientific papers and acknowledges its potential to outperform human authors. The publication "ChatGPT: five priorities for research" by EAM van Dis et al., cited 68 times, outlines the five key areas of focus for further ChatGPT research. Additionally, the visualization highlights the prominence of papers discussing the implications and controversies surrounding ChatGPT, such as "ChatGPT listed as author on research papers: many scientists disapprove" by C Stokel-Walker (67 citations) and "AI bot ChatGPT writes smart essays-should academics worry?" also by C Stokel-Walker (51 citations). Other notable papers include "OpenAI ChatGPT generated literature review: Digital twin in healthcare" by Ö Aydın and E Karaarslan (46 citations), "Collaborating With ChatGPT: Considering the Implications of Generative Artificial Intelligence for Journalism and Media Education" by JV Pavlik (37 citations), "Comparing scientific abstracts generated by ChatGPT to original abstracts using an artificial intelligence output detector, plagiarism detector, and blinded human…" by CA Gao et al. (37 citations), "ChatGPT and other large language models are double-edged swords" by Y Shen et al. (36 citations), and "ChatGPT user experience: Implications for education" by X Zhai (32 citations). These highly cited papers contribute significantly to the understanding and advancement of ChatGPT in various domains.

![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/3c8b49ed-607d-4562-9253-53d6d6282c63)

### 3.Publication Trends

The analysis of publication information related to papers about ChatGPT has yielded valuable insights. Firstly, despite being available for only a short period in 2022, there were already 61 papers published on ChatGPT, indicating a rapid adoption and interest in exploring its capabilities. Moving into 2023, the interest in ChatGPT exploded. By late March, a whopping 809 papers had already been published on the subject, demonstrating that ChatGPT had captured substantial attention within the research community. This remarkable uptick in the number of papers reflects not only the growing utilization of ChatGPT but also the increased investigation into its potential across various domains and applications. 

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/1975aef1-7750-416c-8934-bc4f831af1a0" alt="image" width="500" height="400">

Furthermore, the analysis of publication information based on the number of papers published on various research networks and science databases showcases the distribution of papers across different platforms. The leading research network in terms of ChatGPT papers is arxiv.org, with 104 papers. It is closely followed by papers.ssrn.com with 97 papers. Other prominent platforms include link.springer.com with 52 papers, researchgate.net with 43 papers, and both cureus.com and sciencedirect.com with 25 papers each. Additional platforms that have seen contributions to the ChatGPT research literature include europepmc.org with 21 papers, medrxiv.org with 20 papers, biblioteca.galileo.edu with 19 papers, and thieme-connect.com with 16 papers.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/006a58ae-8397-4303-bddb-8e6bb13e2fe2" alt="image" width="500" height="400">

This analysis highlights the diverse distribution of ChatGPT papers across multiple research networks and science databases, indicating a broad interest in publishing ChatGPT-related research across various platforms. The increase in the number of papers published in 2023 demonstrates the growing significance and research momentum around ChatGPT as a subject of study and exploration.

### 4.Sentiment Analysis 

The sentiment analysis conducted on paper titles related to ChatGPT yielded the following results: 173 positive titles, 525 neutral titles, and 123 negative titles. However, it is important to note that sentiment analysis on academic papers may not always accurately capture the sentiment conveyed within the titles. Academic papers often adopt a neutral and objective tone, focusing on presenting research findings, methodologies, and theoretical frameworks.

Therefore, while the sentiment analysis suggests a distribution of positive, neutral, and negative titles, it is crucial to interpret these results with caution. The sentiment analysis may not accurately reflect the overall sentiment or tone of the papers themselves. It is recommended to delve into the actual content of the papers for a more comprehensive understanding of the researchers' perspectives, findings, and arguments.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/14a68e4a-6556-409c-ba40-5f6758830bb9" alt="image" width="500" height="400">

## Topic Modeling

The LDA topic modeling analysis was performed on the titles of research papers concerning ChatGPT to identify the prominent topics that researchers have been focusing on in their investigations. After evaluating various numbers of topics, it was determined that five topics provided an optimal representation of the dataset. These topics represent key areas of interest and prevailing themes within the ChatGPT research landscape. The top words for each topic, along with their distribution across the documents, are summarized as follows:

Topic 0: this topic encompasses the utilization and exploration of ChatGPT, including its use in various case studies, and its potential applications. It revolves around ChatGPT’s performance. The top 10 words for this topic include 'test,' 'study,' 'chatgpts,' 'graph,' 'explore,' 'information,' 'chatbot,' 'potential,' 'use,' and 'case.'

Topic 1: Ethics and categorization are central to this topic. It delves into the ethical considerations associated with ChatGPT and examines tools and categories related to its deployment. The top words for this topic are 'ethics,' 'tool,' 'category,' 'nature,' 'share,' 'capability,' 'fall,' 'phase,' 'post,' and ‘school.'

Topic 2: Language models and large-scale implementation are the key areas of focus in this topic. It explores the model itself, emphasizing language models and the assessment and evaluation of ChatGPT's performance. The top words in this topic include 'model,' 'language,' 'large,' 'learn,' 'humans,' 'assessment,' 'evaluation,' 'good,' 'teach,' and ‘write.'

Topic 3: Artificial intelligence and its intersection with science and education are the main themes of this topic. It discusses the role of ChatGPT in science, educational settings, and literacy enhancement. The top words for this topic are 'artificial,' 'intelligence,' 'science,' 'education,' 'educational,' 'preliminary,' 'time,' 'literacy,' 'cost,' and ‘challenge.'

Topic 4: This topic focuses on the writing and analysis of scientific papers and considers future implications, research requirements, potential biases, and the academic perspective. The top words associated with this topic include 'write,' 'analysis,' 'scientific,' 'future,' 'research,' 'paper,' 'need,' 'bias,' 'academic,' and ‘perspective.'

The visualization of these topics in the form of word clouds not only aids in understanding the current research focus within the ChatGPT domain but also helps identify science gaps and areas that require further investigation. By analyzing the distribution of top words in each topic, researchers can identify potential areas of research that have received less attention or where additional studies are needed.

![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/fd1574b9-0f33-4d7e-a44b-b57ca0763d19)
![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/1beae316-d62b-49b2-80c2-d2a2581e9031)

To further categorize and label each topic, a meticulous process was undertaken. Initially, the most dominant topic for each paper title was determined. Next, a comparison was made between the top keywords of each topic and the titles in which a particular topic exhibited the highest dominance with a significant contribution percentage. By carefully analyzing these patterns, specific labels were assigned to each topic. The following topic labels were derived:
0: 'Potential Applications and Use Cases of ChatGPT'
1: 'ChatGPT as A Support Tool in Various Fields'
2: 'Implications and Evaluation of ChatGPT'
3: 'Potential and Limitations of ChatGPT in Education and Research'
4: 'ChatGPT and Its Implications for Scientific Writing'
These topic labels serve as concise descriptors, encapsulating the primary themes and areas of focus identified within the collection of paper titles related to ChatGPT.

The distribution of topics across the collection of paper titles revealed varying levels of prominence. Among the analyzed titles, Topic 3 emerged as the most prevalent, with a total of 183 papers primarily focusing on the potential and limitations of ChatGPT in education and research. Topics 4 and 2 closely followed with 180 and 150 documents, respectively, exploring the implications of ChatGPT for scientific writing and the evaluation of its capabilities. Topic 0 and Topic 1 were also significant, with 161 and 148 documents, respectively, delving into the potential applications and use cases of ChatGPT as well as its role as a support tool in various fields. These distributions highlight the diverse research interests and priorities within the realm of ChatGPT.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/0e920f78-2f5b-4f8e-931c-b5d6e41a954d" alt="image" width="500" height="400">

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/d7c1f652-f489-4e01-98c5-e30e540d6cf0" alt="image" width="500" height="400">

___

# Conclusion 

In conclusion, this research project delved into the extensive landscape of ChatGPT research, by scraping and analyzing a comprehensive collection of research papers from Google Scholar. Through a series of analyses, we gained valuable insights into the prevalent themes, research trends, and potential applications of ChatGPT. The findings highlight the practical implementation of ChatGPT as a language model, the focus on generating written content and exploring artificial intelligence aspects, and the growing interest in utilizing ChatGPT in educational and research settings. Moreover, the identification of research gaps provides researchers with opportunities to further expand the knowledge and applications of ChatGPT. Overall, this research contributes to the advancement of ChatGPT, enabling its potential to revolutionize natural language processing and human-computer interactions, while offering guidance for future investigations and innovations in the field.

___

# References 
- Haque, Mubin Ul, et al. (2022). "I think this is the most disruptive technology": Exploring Sentiments of ChatGPT Early Adopters using Twitter Data. 10.48550/arXiv.2212.05856
- Taecharungroj, Viriya. (2023). “What Can ChatGPT Do?” Analyzing Early Reactions to the Innovative AI Chatbot on Twitter. Big Data and Cognitive Computing. 7. 35. 10.3390/bdcc7010035. 
- Scraping Google Scholar: [https://dev.to/dmitryzub/scrape-google-scholar-with-python-32oh](https://dev.to/dmitryzub/scrape-google-scholar-with-python-32oh)
- Topic Modeling: [https://www.linkedin.com/pulse/nlp-a-complete-guide-topic-modeling-latent-dirichlet-sahil-m/](https://www.linkedin.com/pulse/nlp-a-complete-guide-topic-modeling-latent-dirichlet-sahil-m/)
- Twitter Sentiment Analysis about ChatGPT: [https://github.com/hxycorn/Twitter-Sentiment-Analysis-about-ChatGPT/tree/main](https://github.com/hxycorn/Twitter-Sentiment-Analysis-about-ChatGPT/tree/main)

