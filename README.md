**Author**: Christopher Dieck

### Business problem:

The business is trying to see if it can be determined if an individual can
identify as an introvert, an extrovert, or neither based on a series of
questions provided by a questionnaire for the purpose of creating targeted
advertisements.

### Project Goal:

Our goal for this analysis is to identify which features of the dataset are most
important in predicting whether someone can be identified as an introvert or an
extrovert, to discover trends in the data that may be useful, and to build a
machine learning model that can classify if someone is introverted, extroverted,
or neither.


### Data:

- Source: 
    - https://www.kaggle.com/code/yashmehta648/introvert-vs-extrovert/data
- Original Questionnaire: 
    - https://openpsychometrics.org/tests/MIES/development/

#### **Target:**
- Introvert/Extrovert ('IE')
    - Label for being introverted, extroverted, or neither

#### **Features and Context:**
The test contained 91 questions. The questions were presented one at a time in
a random order. For each questions 3 values were recorded:

A - The user's selected response. 1=Disagree, 2=Slightly disagree, 3=Neutral,
4=Slightly agree, 5=Agree

I - The position of the question in the survey.

E - The time elapsed on that question in milliseconds.

The text of the questions were:

{
 "Q1" : "I would never audition to be on a game show."
 
 "Q2" : "I am not much of a flirt."
 
 "Q3" : "I have to psych myself up before I am brave enough to make a phone call."
 
 "Q4" : "I would hate living with room mates."
 
 "Q5" : "I mostly listen to people in conversations."
 
 "Q6" : "I reveal little about myself."
 
 "Q7" : "I spend hours alone with my hobbies."
 
 "Q8" : "I prefer to eat alone."
 
 "Q9" : "I have trouble finding people I want to be friends with."
 
 "Q10" : "I prefer to socialize 1 on 1, than with a group."
 
 "Q11" : "I sometimes speak so quietly people sometimes have trouble hearing me."

 "Q12" : "I do not like to get my picture taken."
 
 "Q13" : "I can keep a conversation going with anyone about anything."

 "Q14" : "I want a huge social circle."
 
 "Q15" : "I talk to people when waiting in lines."
 
 "Q16" : "I act wild and crazy."
 
 "Q17" : "I am a bundle of joy."
 
 "Q18" : "I love excitement."
 
 "Q19" : "I&apos;d like to be in a parade."
 
 "Q20" : "I am a flamboyant person."
 
 "Q21" : "I am good at making impromptu speeches."
 
 "Q22" : "I naturally emerge as a leader."
 
 "Q23" : "I am spontaneous."
 
 "Q24" : "I would enjoy being a sports team coach."
 
 "Q25" : "I have a strong personality."
 
 "Q26" : "I am excited by many different activities."
 
 "Q27" : "I spend most of my time in fantasy worlds."
 
 "Q28" : "I often feel lucky."
 
 "Q29" : "I don't make eye contact when I talk with people."
 
 "Q30" : "I have a monotone voice."
 
 "Q31" : "I am a touchy feely person."

 "Q32" : "I would like to try bungee jumping."
 
 "Q33" : "I tend to be admired by others."
 
 "Q34" : "I make big physical movements whenever I get excited."
 
 "Q35" : "I am brave."
 
 "Q36" : "I am always in the moment."
 
 "Q37" : "I am involved with my community."
 
 "Q38" : "I am good an entertaining children."
 
 "Q39" : "I like formal occasions."
 
 "Q40" : "I would have to be lost for a very long time before asking help."
 
 "Q41" : "I do not care about sports."
 
 "Q42" : "I prefer individual sports to team sports."
 
 "Q43" : "My parents know nothing about my love life."
 
 "Q44" : "I mostly listen to people in conversations."
 
 "Q45" : "I never leave the door to my room open."
 
 "Q46" : "I make a lot of hand motions when I talk."
 
 "Q47" : "I take lots of pictures of my activities."
 
 "Q48" : "When I was a child, I put on fake concerts and plays with my friends."
 
 "Q49" : "I really like dancing."
 
 "Q50" : "I would have difficulty describing myself to someone."
 
 "Q51" : "My life would not make a good story."
 
 "Q52" : "I am hesitant to give suggestions."
 
 "Q53" : "I tire out quickly."
 
 "Q54" : "I never tell people the important things about myself."
 
 "Q55" : "I avoid going to unknown places."
 
 "Q56" : "Going to the doctor is always awkward for me."
 
 "Q57" : "I have not kept up with my old friends over the years."
 
 "Q58" : "I have not been joyful for quite some time."
 
 "Q59" : "I hate to ask for help."
 
 "Q60" : "If I were to die, I would not want there to be a memorial for me."
 
 "Q61" : "I hate shopping."
 
 "Q62" : "I love to do impressions."
 
 "Q63" : "I would be pleased if asked to speak at a funeral."
 
 "Q64" : "I would never go to a dance club."
 
 "Q65" : "I find it very hard to tell people I find them attractive."
 
 "Q66" : "I hate people."
 
 "Q67" : "I was an outcast in school."
 
 "Q68" : "I would enjoy being a librarian."
 
 "Q69" : "I am usually not single."
 
 "Q70" : "I am able to stand up for myself."
 
 "Q71" : "I would go surfing regularly if I lived on a beach."

 "Q72" : "I have wanted to be a stand-up comedian."
 
 "Q73" : "I am a high status person."
 
 "Q74" : "I work out regularly."
 
 "Q75" : "I laugh a lot."
 
 "Q76" : "I like pranks."
 
 "Q77" : "I am happy with my life."
 
 "Q78" : "I am never at a loss for words."
 
 "Q79" : "I feel healthy and vibrant most of the time."
 
 "Q80" : "I love large parties."
 
 "Q81" : "I am quiet around strangers."
 
 "Q82" : "I don&#39;t talk a lot."
 
 "Q83" : "I keep in the background."
 
 "Q84" : "I don&#39;t like to draw attention to myself."
 
 "Q85" : "I have little to say."
 
 "Q86" : "I often feel blue."
 
 "Q87" : "I am not really interested in others."
 
 "Q88" : "I make people feel at ease."
 
 "Q89" : "I don&#39;t mind being the center of attention."
 
 "Q90" : "I start conversations."
 
 "Q91" : "I talk to a lot of different people at parties.
}

After the main question sequence, the following questions were asked on one 
final page (none of the following features were used in machine learning
except for "IE"):

- age: "What is your age in years?"

- gender: "What is your gender?"
    - 1=Male
    - 2=Female
    - 3=Other
- engnat: "Is English your native language?"
    - 1=Yes
    - 2=No

- IE: "Do you identify as either an introvert or extravert?"
    - 1=Yes, introvert
    - 2=Yes, extravert
    - 3=No

On the final page, the users were also asked "Do you give accurate answers and
can we store and use your data for research?". Only those who answered yes
were recorded.

The following were determined from techincal information:

country: user's network location
dateload: the time the user loaded the introduction page
introelapse: the time spent in seconds on the introduction page
testelapse: the time spent in seconds on the test questions
surveyelapse: the time spent in seconds on the final page



**Missing Values**
- There were only two missing values total, and both were in the country 
column. In order to preserve the rest of the data in those rows I imputed 
the missing values with the most frequent value ("US") because it represented
the overwhelming majority of the data.



## Data Preparation Steps for Machine Learning

1. I started by removing the columns that were not going to be useful for any
predictive analysis. These included all of the questions that were asked after
the main questionnaie as well as the technical information, except for the
target column. The final questions were never meant to be used for predictions,
and the technical information was filled with innaccurate values due to people
potentially leaving their computer while taking the questionnaire.

2. I initially set the already ordinal encoded target variable to have string
labels for easier data exploration, so I had to re-ordinal encode them.

3. I split the data into training and testing data to validate my models and be
able to test it on data it has never seen before to prove that it can make
predictions with new data.

4. While all the features have the same range of a value from 1 to 5, I still
used a normalizer to scale the data because the distribution is not a normal
bell curve.

5. Principal Component Analysis (PCA) was used on each model for dimensionality
reduction in order to improve model speed. It was useful and improved the score
on one of the models, but it sacrified accuracy on the others.

## Model Development
- The following three machine learning models were used to see which model can
best classify if someone is an introvert, an extrovert, or neither:
    - Decision Tree Classifier
    - K Nearest Neighbors (KNN) Classifer
    - Logistic Regression Classifier
    
- Each model was evaluated using classification reports that indentified
precision, recall, f1-scores, and accuracy to get an overview of the model
performance and the amount of type 1 and type 2 error. Each was also tested
once more using

- There were several aspects of the dataset that potentially made it difficult 
to produce a model that could reach an accuracy above 75%.

    - A correlational heatmap showed that a large majority of the features did
    not have much correlation to the target variable
    - The dataset provided was relatively small with roughly 7,200 entries and
    few relevant features.

## Results

### Decision Tree Classification Report and Confusion Matrix
<img src=https://github.com/ChrisDieck/Project-2/blob/main/dec_tree_cr.png>
<img src=https://github.com/ChrisDieck/Project-2/blob/main/dec_tree_cm.png>

- As shown, the decision tree model performed okay with an accuracy of 
71.2% on the training set and 71.8% on the testing set. A redeeming quality is
that it has a fairly high recall score of 92% for identifying introverts.

### KNN Classification Report and Confusion Matrix
<img src=https://github.com/ChrisDieck/Project-2/blob/main/knn_cr.png>
<img src=https://github.com/ChrisDieck/Project-2/blob/main/knn_cm.png>

- The KNN model performed slightly better with a 73.1% accuracy on the training
set and 73.4% on the testing set. Although only a slight increase in accuracy,
it had a great improvement with recall for detecting extroverts by about 20%,
but it lowered by 4% for those who identify as neither.

### KNN with PCA Classification Report and Confusion Matrix
<img src=https://github.com/ChrisDieck/Project-2/blob/main/knn_pca_cr.png>
<img src=https://github.com/ChrisDieck/Project-2/blob/main/knn_pca_cm.png>

- The KNN model using PCA performed about 1% better than the base model, which
may not be worth losing the ability to identify the features. It also performed
slightly worse on recall scores, but slightly better for precision scores.

### Logistic Regression Classification Report and Confusion Matrix
<img src=https://github.com/ChrisDieck/Project-2/blob/main/logreg_cr.png>
<img src=https://github.com/ChrisDieck/Project-2/blob/main/logreg_cm.png>

- The logistic regression model performed the best out of all the models with
an accuracy of 74.9%. Also, it had the highest recall scores for identifying
extroverts or ambiverts(neither), but it performed slightly worse for 
identifying introverts specifically. On the other hand, it also performed the 
best on precision scores across the board.

## Most Useful Data Visualizations
### Correlational Heatmap
<img src=https://github.com/ChrisDieck/Project-2/blob/main/p2_heatmap.png>

- This heatmap shows that there is a lot of multicollinearity among the questions
(shown by the brighter yellow squares). One of them is question 44 and question
5. Another is question 44 and question 6. Both have higher correlations, and 
many others have moderate correlations among each other.

- By looking at the bottom row, we can see that there may be some mild to
moderate correlations between the target variable (IE) and the questions. It
seems that questions 13-26 have the highest correlations to 'IE', along with
questions 31-39, 47-49, 80, and 89-91.

### Bar Chart Showing the % of People That Can Keep a Coversation Going
<img src=https://github.com/ChrisDieck/Project-2/blob/main/q13a_and_IE.png>

### Bar Chart Showing the % of People That Like to Start Conversations
<img src=https://github.com/ChrisDieck/Project-2/blob/main/Q90A_and_IE.png>

Analysis of Bar Charts:

Both of these barplots show that people who identify as introverted typically
seem to either dislike or avoid starting or continuing conversations, especially
with people they do not know.

The answers to these questions are also highly correlated to other questions 
regarding situations where the individual would be around talkative people, 
such as at parties. Across all of these types of questions, those who identify
as introverts tend to not be as talkative as those who identify as extroverted.

Using these questions to give a quick idea on whether or not someone is an 
introvert or extrovert could be extremely useful for someone who is trying to
market towards introverts or extroverts specifically through having more
information on how these types of people are.

It is also worth mentioning that those who identify as "Neither" (which could be
described as an ambivert, or someone who is in between introverted and
extroverted) answered fairly evenly across the board, as expected.

## Final Model

Based on these reports, the logistic regression model had the highest accuracy 
at roughly 75%, followed by the KNN model with PCA at 74%, then the KNN model
without PCA and the decision tree model at 71%.

Of these models, I would consider the logistic regression model to be the best
simply because it has the highest accuracy. While having false positives and
negatives are not very important for this particular situation, having less
false negatives is slightly better because (in the case of deciding how to
market to a particular individual) false negatives mean we are missing
opportunities to advertise to someone who would respond well to the targeted
advertisement.

A false positive, on the other hand, means we might send an advertisement to
someone who might not be as likely to care about it, but there is a chance they
could still like the product. In short, the cost of taking action is low, while
the cost of missing the opportunity can be much higher.

## Recommendations:
- Using the logistic regression model, a business can make predictions on
classifying whether someone is introverted, extroverted, or neither.

- Using this information, the company can make changes to who they advertise to
to ultimately increase sales and/or not waste as many resources on who they
advertise to. It can also be used to see what category the companies main
audience tends to fall into, and then adjustments can be made accordingly
depending on the desired outcomes.

- Additionally, it is highly recommended to disclude the same features that I
discluded due to the reasons listed in the method section above. They are not
suitable predictions and will most likely add a lot of noise.


## Limitations & Next Steps
The biggest limitations with this project was having features that were
difficult to visualize in a meaningful way due the numerous amount of them and
how they are formatted. Aside from that, many of them seem to not have much of a
great correlation with the target, which is a lot of extra data collected that
was unnecessary.

The next steps should be as follows:

1. Further experimentation may prove to be useful for hypertuning the model,
especially regarding increasing the recall score for the Extrovert and Neither
classes.

2. I could try using a boosting method such as XGBoost to potentially increase
performance.

3. I should cross validate the model to ensure that it can perform just as well
on multiple test sets aside from the main one I used.

### For further information

For any additional questions, please contact christopherjdieck@gmail.com
