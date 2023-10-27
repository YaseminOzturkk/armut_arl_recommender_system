
#########################
# Business Problem
#########################

# Armut, which is Turkey's largest online service platform, brings together service providers and those who want to
# receive services. It enables easy access to services such as cleaning, renovation, and transportation with just a few
# touches on a computer or smartphone. An association rule learning-based product recommendation system is desired
# to be created using the dataset containing users who received services and the categories of services they received.


#########################
# Story of the Dataset
#########################
# The dataset consists of the services customers receive and the categories of these services. Date and time of each
# service received contains information.

# UserId: Customer ID
# ServiceId: They are anonymized services belonging to each category. (Example: Upholstery washing service under
# Cleaning category) A ServiceId can be found under different categories and represents different services under
# different categories. (Example: The service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while the
# service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: They are anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


#########################
# TASK 1: Preparing the Data
#########################

# Step 1: Read to armut_data.csv file.


df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.head
df.info()
df.describe().T
df.isnull().sum()
df.shape


Step 2: ServiceId represents a different service for each CategoryId. Combine ServiceId and CategoryId
# with "_" to create a new variable to represent the services.

df["Services"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()

Step 3: The dataset consists of services purchased by customers with the date and time of purchase, but there is
# no basket definition (invoice, etc.). In order to apply Association Rule Learning, a basket definition needs to be
# created, which represents the services purchased by each customer on a monthly basis. For example, customer with
# ID 7256 has a basket consisting of services 9_4 and 46_4 purchased in August 2017, and a different basket consisting
# of services 9_4 and 38_4 purchased in October 2017. The baskets should be identified with a unique ID. To do this,
# first create a new date variable that only includes the year and month. Then combine UserId and the new date variable
# using "_" and assign it to a new variable named BasketId.


df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df.info()

# dt.to_period metodu,tarih veya zaman bilgisini dönemlere dönüştürmek için kullanılır.

df["NewDate"] = df["CreateDate"].dt.to_period("M")
df.head()

df["BasketId"] = df["UserId"].astype(str) + "_" + df["NewDate"].astype(str)
df.head()

#########################
# TASK 2: Create Association Rules
#########################

#  Step 1: Create a pivot table with BasketId values in rows and Service values in columns.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketId
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


# First break it down by BasketId, then by Services, how many purchased Services are in the cart:
# CategoryId is inside Services, so we can also get the count of Services.


# unstack: The unstack() function is used to pivot after the groupby operation, that is, to pass the "Services" variable to columns.
# We want missing values to be written as 0 and filled values as 1.

# After the unstack() operation, the fillna(0) method is used to fill the empty spaces with 0.


# We should write 1 for any number greater than 0 and 0 for others.
# Because we expect a special matrix structure that is more measurable and on which we can perform analytical operations.
# Here we use the applymap() function. Because applymap() traverses all observations (in all rows and columns).

apriori_df = df.groupby(["BasketId", "Services"])["CategoryId"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# Step 2: Create association rules.

# First, let's find the Support values, or probabilities, of all possible product combinations with the apriori() function.
# Here min_support is the minimum Support value we want to determine, threshold.
# If we want to use the names of the variables in the data set we want to use, use_colnames=True is added.

frequent_itemsets = apriori(apriori_df.astype("bool"),
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

# We currently have possible Services or Services pairs and their corresponding support values.
# There are no possible values below 0.01 here because we gave the minimum support value (threshold value) as 0.01.
# These are the probabilities of each service. What we need are association rules.



# For the association rules we need, we will use this data with the association_rules() method and extract the association rules from it:

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()

# antecedents: Previous Service
#consequences: Second Service
# antecedent support: Probability of observing First Service alone
# consequent support: Probability of observing the Second Service alone
# support: Possibility of two Services appearing together
# confidence: The probability of receiving the second service when the first service is received
# lift: Indicates how many times the probability of receiving a second Service increases when one Service is purchased.
# leverage: similar to lift. It tends to prioritize values with high support, so it has a slight bias.
# conviction: Expected frequency of one Service without another

# Step 3: Step 6: Using the arl_recommender function, recommend a service to a user who has received the last

def arl_recommender(rules_df, service_id, rec_num=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, service in enumerate(sorted_rules["antecedents"]):
        for j in list(service):
            if j == service_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_num]


# service_id: The service id for which we want to make a recommendation.
# rec_num: Returns the desired number of recommendation services.
# First of all, we listed the rules from largest to smallest according to fiber in order to capture the first most compatible product.
# (This order may also be based on confidence, depending on preference.)
# An empty list is created for the products to be recommended.
# We use the enumerate() method according to the service that comes first in the sorted rules.
# In the second loop, the services will be browsed. If the service for which advice is requested is caught,
# The index information was kept by i. This adds the consequents value in the index information to the recommendation_list.
# [0] is added to bring the first service it sees.

arl_recommender(rules,"2_0",3)
df.head

# Note: As the number of recommended services increases, the values of other corresponding services in the relevant statistics will be lower.
