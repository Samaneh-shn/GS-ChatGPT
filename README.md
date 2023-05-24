# "Insights into ChatGPT Research: A Comprehensive Analysis"

The objective of this project was to conduct a comprehensive analysis of research papers related to ChatGPT by scraping data from Google Scholar. The dataset obtained consisted of numerous research papers covering various aspects of ChatGPT, and the analysis aimed to gain insights into the research landscape surrounding this topic.

<img src="https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/632cfc65-50e9-40c0-8618-d11d02e75ff6" alt="image" width="400" height="300">

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

In this project, we delve into the ChatGPT research landscape, aiming to provide a comprehensive analysis of the prevailing themes, research trends, and potential areas of exploration within this domain. By conducting extensive data collection, exploratory data analysis, and topic modeling, we aim to shed light on the key areas of interest, identify potential research gaps, and uncover insights that can guide future studies.
Through our research, we aim to contribute to the growing body of knowledge surrounding ChatGPT, providing researchers, practitioners, and decision-makers with valuable insights into the advancements, challenges, and opportunities within the realm of conversational AI. By understanding the current state of research, we can further harness the potential of ChatGPT, pushing the boundaries of human-computer interactions and paving the way for more intelligent and intuitive AI systems.

___

# Methodology 

## Data Collection

The initial step involved data collection, wherein we leveraged ScraperAPI to scrape Google Scholar and amass a dataset of approximately 1000 research papers related to "ChatGPT". ScraperAPI proved instrumental in handling proxies, browsers, and CAPTCHAs, enabling us to retrieve the HTML content of web pages through a straightforward API call. Our search query was specifically tailored to "ChatGPT", and we extracted various data points including paper titles, links, paper snippets, citation details, and publication information from the first 100 pages of Google Scholar on March 29, 2023. To circumvent potential breakdowns or IP blocks, we executed the code iteratively for a total of 10 times, scraping only 10 pages per iteration. Ultimately, we concatenated all the extracted data from the CSV files into a single dataset.

## Data Preprocessing

In the preprocessing phase, we employed various techniques to enhance the quality of our data. Firstly, we utilized the detect function from the langdetect library to filter out non-English titles, ensuring that our analysis focused solely on English-language papers. Next, we employed regular expressions (regex) to eliminate occurrences of [HTML][HTML] or [PDF][PDF] at the beginning of paper titles, as they were artifacts of the scraping process. To further refine the data, we leveraged several components from the Natural Language Toolkit (NLTK) library, including the removal of punctuation and stop words using word_tokenize and stopwords functions, respectively. 
Additionally, we eliminated the frequently occurring term "ChatGPT" from the titles, as it represented the overarching topic of our analysis. Furthermore, we applied lemmatization and stemming techniques using the WordNetLemmatizer and SnowballStemmer functions to normalize the remaining words, reducing them to their base form. These preprocessing steps ensured that our dataset contained relevant and standardized textual information for subsequent analysis. In addition to the paper titles, we also applied the same preprocessing steps to the paper snippets.

### EDA

For the exploratory data analysis (EDA), we performed several key steps to gain insights from the cleaned text data. First, we combined all the cleaned titles and snippets into a single list. This allowed us to work with a unified dataset for further analysis.

Next, we utilized the Counter module from the collections library to count the frequency of each word in the cleaned text. By examining the word frequencies, we were able to identify the most common terms and gain a general understanding of the content.
To visualize the top 20 words, we employed the LetsPlot library, using the setup_html() function to generate an interactive bar chart. This visualization provided a clear representation of the most frequent words, enabling us to identify prominent themes or topics within the ChatGPT research domain.

Furthermore, we utilized the WordCloud module from the wordcloud library to create a visually appealing word cloud based on the word counts. This allowed us to observe the relative prominence of different words in a visually striking manner.

To gain insights into the citation counts of the papers, we utilized the search function from the regex library to extract the citation count information from the scraped data. We then employed the sns module from the seaborn library to visualize the top 10 cited papers, providing a glimpse into the most influential works in the field.

moreover, we employed the str.extract function to extract publication information such as the publication year and the names of the research networks or science databases on which the papers were published. This information allowed us to analyze the distribution of publications across different platforms and track the temporal trends of ChatGPT research.

lastly, To gain insights into the sentiment expressed in the text data, we utilized the TextBlob module from the textblob library for sentiment analysis. This allowed us to assess the polarity and subjectivity of the paper titles and snippets, providing an understanding of the overall sentiment associated with the ChatGPT research. 

By performing these EDA techniques, we gained valuable insights into the cleaned text data, identified significant words and topics, explored citation patterns, and examined publication information. These findings laid the foundation for further analysis and interpretation of the ChatGPT research landscape.


## Data Modeling

For the topic modeling phase, we employed several libraries to perform LDA (Latent Dirichlet Allocation) topic modeling on the collected data. The libraries used included gensim, pyLDAvis.gensim_models, and corpora from gensim.
To begin, we created a dictionary and a Bag-of-Words (BoW) corpus using the gensim library. These representations allowed us to transform the textual data into a numerical format suitable for topic modeling analysis.
To determine the optimal number of topics, we followed an iterative approach. We built multiple LDA models with different values of the number of topics (k) and assessed the coherence value for each model. The coherence value serves as an indicator of the interpretability and meaningfulness of the topics generated. Typically, selecting a value of k where the coherence score reaches its peak and exhibits a diminishing growth pattern leads to the most meaningful topics. However, if the same keywords are repeated across multiple topics, it suggests that k may be too large.
To automate this process, we developed a function that takes in the Gensim dictionary, Gensim corpus, input texts, and a range of potential topics. This function returns a list of models and their corresponding coherence values, aiding in the selection of the optimal number of topics.
In our case, we created ten LDA models with different numbers of topics and evaluated the coherence scores for each model. Visualizing the coherence scores for different numbers of topics (ranging from 2 to 10), we observed that five topics yielded the highest coherence score of 0.593.

![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/7fdc27ec-c1d3-409b-ba6b-1e3fab8de588)

Furthermore, we focused on refining the hyperparameters to improve the model's performance. We specifically looked at the hyperparameters alpha and beta, which respectively control the document-topic density and word-topic density. After experimentation, we found that setting both alpha and beta to symmetric values resulted in the best performance, with a coherence score improvement of about 2% over the baseline model.

![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/6335c6b7-bc82-42c9-8424-26b283b16460)

With the optimal hyperparameters determined, we created the final LDA model with a coherence score of 0.593. To visualize the topics and their distribution, we utilized pyLDAvis.gensim_models and the LDAvis_prepared function. This allowed us to generate an interactive visualization of the topics and their corresponding keywords, facilitating a better understanding of the relationships and characteristics of each topic.

![image](https://github.com/Samaneh-shn/GS-ChatGPT/assets/120117013/2fc7b1d5-741a-41bd-8f37-2b2ef8807f3d)

By leveraging these techniques and libraries, we successfully performed LDA topic modeling on the ChatGPT research papers, ultimately obtaining a coherent and interpretable set of topics that encapsulated the main areas of focus within the dataset.



