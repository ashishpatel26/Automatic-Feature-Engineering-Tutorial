
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Introduction: Automated Feature Engineering<a class="anchor-link" href="#Introduction:-Automated-Feature-Engineering">¶</a>
</h1>
<p>In this notebook, we will look at an exciting development in data science: automated feature engineering. A machine learning model can only learn from the data we give it, and making sure that data is relevant to the task is one of the most crucial steps in the machine learning pipeline (this is made clear in the excellent paper <a href="https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf">"A Few Useful Things to Know about Machine Learning"</a>).</p>
<p>However, manual feature engineering is a tedious task and is limited by both human imagination - there are only so many features we can think to create - and by time - creating new features is time-intensive. Ideally, there would be an objective method to create an array of diverse new candidate features that we can then use for a machine learning task. This process is meant to not replace the data scientist, but to make her job easier and allowing her to supplement domain knowledge with an automated workflow.</p>
<p>In this notebook, we will walk through an implementation of using <a href="https://www.featuretools.com/">Featuretools</a>, an open-source Python library for automatically creating features with relational data (where the data is in structured tables). Although there are now many efforts working to enable automated model selection and hyperparameter tuning, there has been a lack of automating work on the feature engineering aspect of the pipeline. This library seeks to close that gap and the general methodology has been proven effective in both <a href="https://github.com/HDI-Project/Data-Science-Machine">machine learning competitions with the data science machine</a> and <a href="https://www.featurelabs.com/blog/predicting-credit-card-fraud/">business use cases</a>.</p>
<h2>Dataset<a class="anchor-link" href="#Dataset">¶</a>
</h2>
<p>To show the basic idea of featuretools we will use an example dataset consisting of three tables:</p>
<ul>
<li>
<code>clients</code>: information about clients at a credit union</li>
<li>
<code>loans</code>: previous loans taken out by the clients</li>
<li>
<code>payments</code>: payments made/missed on the previous loans</li>
</ul>
<p>The general problem of feature engineering is taking disparate data, often distributed across multiple tables, and combining it into a single table that can be used for training a machine learning model. Featuretools has the ability to do this for us, creating many new candidate features with minimal effort. These features are combined into a single table that can then be passed on to our model.</p>
<p>First, let's load in the data and look at the problem we are working with.</p>

</div>


```python
# Run this if featuretools is not already installed
# !pip install -U featuretools
```


```python
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')
```


```python
client = pd.read_csv("Clients.csv", parse_dates=['joined'])
```


```python
loans = pd.read_csv("loans.csv", parse_dates=['loan_start','loan_end'])
```


```python
payment = pd.read_csv("payments.csv", parse_dates=["payment_date"])
```


```python
client.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
    </tr>
  </tbody>
</table>
</div>




```python
loans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>loan_type</th>
      <th>loan_amount</th>
      <th>repaid</th>
      <th>loan_id</th>
      <th>loan_start</th>
      <th>loan_end</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>home</td>
      <td>13672</td>
      <td>0</td>
      <td>10243</td>
      <td>2002-04-16</td>
      <td>2003-12-20</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46109</td>
      <td>credit</td>
      <td>9794</td>
      <td>0</td>
      <td>10984</td>
      <td>2003-10-21</td>
      <td>2005-07-17</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46109</td>
      <td>home</td>
      <td>12734</td>
      <td>1</td>
      <td>10990</td>
      <td>2006-02-01</td>
      <td>2007-07-05</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46109</td>
      <td>cash</td>
      <td>12518</td>
      <td>1</td>
      <td>10596</td>
      <td>2010-12-08</td>
      <td>2013-05-05</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46109</td>
      <td>credit</td>
      <td>14049</td>
      <td>1</td>
      <td>11415</td>
      <td>2010-07-07</td>
      <td>2012-05-21</td>
      <td>3.13</td>
    </tr>
  </tbody>
</table>
</div>




```python
payment.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>payment_amount</th>
      <th>payment_date</th>
      <th>missed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10243</td>
      <td>2369</td>
      <td>2002-05-31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10243</td>
      <td>2439</td>
      <td>2002-06-18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10243</td>
      <td>2662</td>
      <td>2002-06-29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10243</td>
      <td>2268</td>
      <td>2002-07-20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10243</td>
      <td>2027</td>
      <td>2002-07-31</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




# Manual Feature Engineering Examples
Let's show a few examples of features we might make by hand. We will keep this relatively simple to avoid doing too much work! First we will focus on a single dataframe before combining them together. In the clients dataframe, we can take the month of the joined column and the natural log of the income column. Later, we see these are known in featuretools as transformation feature primitives because they act on column in a single table.


```python
#create client month 
client["join_month"] = client["joined"].dt.month

# create a log of income
client["log_income"] = np.log(client["income"])

client.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
    </tr>
  </tbody>
</table>
</div>




To incorporate information about the other tables, we use the df.groupby method, followed by a suitable aggregation function, followed by df.merge. For example, let's calculate the average, minimum, and maximum amount of previous loans for each client. In the terms of featuretools, this would be considered an aggregation feature primitive because we using multiple tables in a one-to-many relationship to calculate aggregation figures (don't worry, this will be explained shortly!).


```python
stats = loans.groupby("client_id")["loan_amount"].agg(["mean","max","min"])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']
stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_loan_amount</th>
      <th>max_loan_amount</th>
      <th>min_loan_amount</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>7963.950000</td>
      <td>13913</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>7270.062500</td>
      <td>13464</td>
      <td>1164</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>7824.722222</td>
      <td>14865</td>
      <td>2389</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>7125.933333</td>
      <td>14593</td>
      <td>653</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>9813.000000</td>
      <td>14837</td>
      <td>2778</td>
    </tr>
  </tbody>
</table>
</div>




```python
client.merge(stats, left_on="client_id", right_index=True, how="left").head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client_id</th>
      <th>joined</th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>mean_loan_amount</th>
      <th>max_loan_amount</th>
      <th>min_loan_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46109</td>
      <td>2002-04-16</td>
      <td>172677</td>
      <td>527</td>
      <td>4</td>
      <td>12.059178</td>
      <td>8951.600000</td>
      <td>14049</td>
      <td>559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49545</td>
      <td>2007-11-14</td>
      <td>104564</td>
      <td>770</td>
      <td>11</td>
      <td>11.557555</td>
      <td>10289.300000</td>
      <td>14971</td>
      <td>3851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41480</td>
      <td>2013-03-11</td>
      <td>122607</td>
      <td>585</td>
      <td>3</td>
      <td>11.716739</td>
      <td>7894.850000</td>
      <td>14399</td>
      <td>811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46180</td>
      <td>2001-11-06</td>
      <td>43851</td>
      <td>562</td>
      <td>11</td>
      <td>10.688553</td>
      <td>7700.850000</td>
      <td>14081</td>
      <td>1607</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25707</td>
      <td>2006-10-06</td>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>13913</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39505</td>
      <td>2011-10-14</td>
      <td>153873</td>
      <td>610</td>
      <td>10</td>
      <td>11.943883</td>
      <td>7424.050000</td>
      <td>14575</td>
      <td>904</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32726</td>
      <td>2006-05-01</td>
      <td>235705</td>
      <td>730</td>
      <td>5</td>
      <td>12.370336</td>
      <td>6633.263158</td>
      <td>14802</td>
      <td>851</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35089</td>
      <td>2010-03-01</td>
      <td>131176</td>
      <td>771</td>
      <td>3</td>
      <td>11.784295</td>
      <td>6939.200000</td>
      <td>13194</td>
      <td>773</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35214</td>
      <td>2003-08-08</td>
      <td>95849</td>
      <td>696</td>
      <td>8</td>
      <td>11.470529</td>
      <td>7173.555556</td>
      <td>14767</td>
      <td>667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>48177</td>
      <td>2008-06-09</td>
      <td>190632</td>
      <td>769</td>
      <td>6</td>
      <td>12.158100</td>
      <td>7424.368421</td>
      <td>14740</td>
      <td>659</td>
    </tr>
  </tbody>
</table>
</div>



We could go further and include information about payments in the clients dataframe. To do so, we would have to group payments by the **loan_id, merge** it with the **loans**, group the resulting dataframe by the **client_id**, and then merge it into the clients dataframe. This would allow us to include information about previous payments for each client.

Clearly, this process of manual feature engineering can grow quite tedious with many columns and multiple tables and I certainly don't want to have to do this process by hand! Luckily, featuretools can automatically perform this entire process and will create more features than we would have ever thought of. Although I love pandas, there is only so much manual data manipulation I'm willing to stand!

<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Featuretools<a class="anchor-link" href="#Featuretools">¶</a>
</h1>
<p>Now that we know what we are trying to avoid (tedious manual feature engineering), let's figure out how to automate this process. Featuretools operates on an idea known as <a href="https://docs.featuretools.com/api_reference.html#deep-feature-synthesis">Deep Feature Synthesis</a>. You can read the <a href="http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf">original paper here</a>, and although it's quite readable, it's not necessary to understand the details to do automated feature engineering. The concept of Deep Feature Synthesis is to use basic building blocks known as feature primitives (like the transformations and aggregations done above) that can be stacked on top of each other to form new features. The depth of a "deep feature" is equal to the number of stacked primitives.</p>
<p>I threw out some terms there, but don't worry because we'll cover them as we go. Featuretools builds on simple ideas to create a powerful method, and we will build up our understanding in much the same way.</p>
<p>The first part of Featuretools to understand <a href="https://docs.featuretools.com/loading_data/using_entitysets.html#adding-entities">is an <code>entity</code></a>. This is simply a table, or in <code>pandas</code>, a <code>DataFrame</code>. We corral multiple entities into a <a href="https://docs.featuretools.com/loading_data/using_entitysets.html">single object called an <code>EntitySet</code></a>. This is just a large data structure composed of many individual entities and the relationships between them.</p>
<h2>EntitySet<a class="anchor-link" href="#EntitySet">¶</a>
</h2>
<p>Creating a new <code>EntitySet</code> is pretty simple:</p>

</div>
</div>


```python
es = ft.EntitySet(id = 'client')
```

# Entities
An entity is simply a table, which is represented in **Pandas as a dataframe**. Each **entity** must have a uniquely identifying **column**, known as an **index**. For the **clients dataframe,** this is the **client_id** because **each id only appears once in the clients data.** In the **loans dataframe, client_id** is not an **index** because each id might appear more than once. The index for this dataframe is instead **loan_id.**

When we create an **entity(Table) in featuretools**, we have to ***identify which column of the dataframe is the index.*** *If the data **does not have a unique index** we can tell **featuretools to make an index for the entity by passing in make_index = True*** and specifying a **name for the index.** If the data also has a uniquely identifying **time index,** we can pass that in as the **time_index parameter.**

**Featuretools** will automatically infer the **variable types (numeric, categorical, datetime)** of the columns in our data, but *we can also pass in specific datatypes to override this behavior.* As an example, ***even though the repaid column in the loans dataframe is represented as an integer,** we can tell featuretools that this is a ***categorical feature since it can only take on two discrete values.*** This is done using an **integer with the variables as keys and the feature types as values.**

In the code below we create the **three entities and add them to the EntitySet.** The syntax is relatively straightforward with a few notes: for **the payments dataframe we need to make an index**, **for the loans dataframe, we specify that repaid is a categorical variable**, and for the **payments dataframe,** we specify that missed is a **categorical feature.**


```python
# Create an entity from the client dataframe# Create 
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id="client",dataframe=client,index='client_id',time_index='joined')
es
```




    Entityset: client
      Entities:
        client [Rows: 25, Columns: 6]
      Relationships:
        No relationships




```python
# Create an entity from the loans dataframe# Create 
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id="loans", dataframe=loans,
                             variable_types = {'repaid': ft.variable_types.Categorical},
                            index='loan_id',time_index="loan_start")
es
```




    Entityset: client
      Entities:
        client [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
      Relationships:
        No relationships




```python
# Create an entity from the payments dataframe
# This does not yet have a unique index
es = es.entity_from_dataframe(entity_id='payment', 
                              dataframe=payment, 
                              variable_types = {'missed':ft.variable_types.Categorical},
                              make_index = True, 
                              index = 'payment_id',
                              time_index = 'payment_date')
es
```




    Entityset: client
      Entities:
        client [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payment [Rows: 3456, Columns: 5]
      Relationships:
        No relationships




```python
es['payment']
```




    Entity: payment
      Variables:
        payment_id (dtype: index)
        loan_id (dtype: numeric)
        payment_amount (dtype: numeric)
        payment_date (dtype: datetime_time_index)
        missed (dtype: categorical)
      Shape:
        (Rows: 3456, Columns: 5)



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2>Relationships<a class="anchor-link" href="#Relationships">¶</a>
</h2>
<p>After defining the entities (tables) in an <code>EntitySet</code>, we now need to tell featuretools <a href="https://docs.featuretools.com/loading_data/using_entitysets.html#adding-a-relationship">how they are related with a relationship</a>. The most intuitive way to think of relationships is with the parent to child analogy: a parent-to-child relationship is one-to-many because for each parent, there can be multiple children. The <code>client</code> dataframe is therefore the parent of the <code>loans</code> dataframe because while there is only one row for each client in the <code>client</code> dataframe, each client may have several previous loans covering multiple rows in the <code>loans</code> dataframe. Likewise, the <code>loans</code> dataframe is the parent of the <code>payments</code> dataframe because each loan will have multiple payments.</p>
<p>These relationships are what allow us to group together datapoints using aggregation primitives and then create new features. As an example, we can group all of the previous loans associated with one client and find the average loan amount. We will discuss the features themselves more in a little bit, but for now let's define the relationships.</p>
<p>To define relationships, we need to specify the parent variable and the child variable. This is the variable that links two entities together. In our example, the <code>client</code> and <code>loans</code> dataframes are linked together by the <code>client_id</code> column. Again, this is a parent to child relationship because for each <code>client_id</code> in the parent <code>client</code> dataframe, there may be multiple entries of the same <code>client_id</code> in the child <code>loans</code> dataframe.</p>
<p>We codify relationships in the language of featuretools by specifying the parent variable and then the child variable. After creating a relationship, we add it to the <code>EntitySet</code>.</p>

</div>
</div>


```python
# Relationship between clients and previous loans
r_client_previous = ft.Relationship(es['client']['client_id'],
                                    es['loans']['client_id'])
# Add the relationship to the entity set
es = es.add_relationship(r_client_previous)
```


```python
es
```




    Entityset: client
      Entities:
        client [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payment [Rows: 3456, Columns: 5]
      Relationships:
        loans.client_id -> client.client_id



<div class="text_cell_render border-box-sizing rendered_html">
<p>The relationship has now been stored in the entity set. The second relationship is between the <code>loans</code> and <code>payments</code>. These two entities are related by the <code>loan_id</code> variable.</p>

</div>


```python
# Relationship between previous loans and previous payments# Relati 
r_payments = ft.Relationship(es['loans']['loan_id'],es['payment']['loan_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_payments)

es
```




    Entityset: client
      Entities:
        client [Rows: 25, Columns: 6]
        loans [Rows: 443, Columns: 8]
        payment [Rows: 3456, Columns: 5]
      Relationships:
        loans.client_id -> client.client_id
        payment.loan_id -> loans.loan_id



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We now have our entities in an entityset along with the relationships between them. We can now start to making new features from all of the tables using stacks of feature primitives to form deep features. First, let's cover feature primitives.</p>
<h2>Feature Primitives<a class="anchor-link" href="#Feature-Primitives">¶</a>
</h2>
<p>A <a href="https://docs.featuretools.com/automated_feature_engineering/primitives.html">feature primitive</a> a at a very high-level is an operation applied to data to create a feature. These represent very simple calculations that can be stacked on top of each other to create complex features. Feature primitives fall into two categories:</p>
<ul>
<li>
<strong>Aggregation</strong>: function that groups together child datapoints for each parent and then calculates a statistic such as mean, min, max, or standard deviation. An example is calculating the maximum loan amount for each client. An aggregation works across multiple tables using relationships between tables.</li>
<li>
<strong>Transformation</strong>: an operation applied to one or more columns in a single table. An example would be extracting the day from dates, or finding the difference between two columns in one table.</li>
</ul>
<p>Let's take a look at feature primitives in featuretools. We can view the list of primitives:</p>

</div>
</div>


```python
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mean</td>
      <td>aggregation</td>
      <td>Computes the average value of a numeric feature.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mode</td>
      <td>aggregation</td>
      <td>Finds the most common element in a categorical feature.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>num_unique</td>
      <td>aggregation</td>
      <td>Returns the number of unique categorical variables.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>trend</td>
      <td>aggregation</td>
      <td>Calculates the slope of the linear trend of variable overtime.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>std</td>
      <td>aggregation</td>
      <td>Finds the standard deviation of a numeric feature ignoring null values.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>skew</td>
      <td>aggregation</td>
      <td>Computes the skewness of a data set.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>avg_time_between</td>
      <td>aggregation</td>
      <td>Computes the average time between consecutive events.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>any</td>
      <td>aggregation</td>
      <td>Test if any value is 'True'.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>num_true</td>
      <td>aggregation</td>
      <td>Finds the number of 'True' values in a boolean.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>count</td>
      <td>aggregation</td>
      <td>Counts the number of non null values.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sum</td>
      <td>aggregation</td>
      <td>Counts the number of elements of a numeric or boolean feature.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>min</td>
      <td>aggregation</td>
      <td>Finds the minimum non-null value of a numeric feature.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>last</td>
      <td>aggregation</td>
      <td>Returns the last value.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>time_since_last</td>
      <td>aggregation</td>
      <td>Time since last related instance.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>max</td>
      <td>aggregation</td>
      <td>Finds the maximum non-null value of a numeric feature.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>median</td>
      <td>aggregation</td>
      <td>Finds the median value of any feature with well-ordered values.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>percent_true</td>
      <td>aggregation</td>
      <td>Finds the percent of 'True' values in a boolean feature.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>all</td>
      <td>aggregation</td>
      <td>Test if all values are 'True'.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>n_most_common</td>
      <td>aggregation</td>
      <td>Finds the N most common elements in a categorical feature.</td>
    </tr>
  </tbody>
</table>
</div>




```python
primitives[primitives['type'] == 'transform']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>longitude</td>
      <td>transform</td>
      <td>Returns the second value on the tuple base feature.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>negate</td>
      <td>transform</td>
      <td>Creates a transform feature that negates a feature.</td>
    </tr>
    <tr>
      <th>21</th>
      <td>not</td>
      <td>transform</td>
      <td>For each value of the base feature, negates the boolean value.</td>
    </tr>
    <tr>
      <th>22</th>
      <td>minute</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the minute.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>years</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of years.</td>
    </tr>
    <tr>
      <th>24</th>
      <td>cum_min</td>
      <td>transform</td>
      <td>Calculates the min of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
    <tr>
      <th>25</th>
      <td>time_since</td>
      <td>transform</td>
      <td>Calculates time since the cutoff time.</td>
    </tr>
    <tr>
      <th>26</th>
      <td>hour</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the hour.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>day</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the day.</td>
    </tr>
    <tr>
      <th>28</th>
      <td>absolute</td>
      <td>transform</td>
      <td>Absolute value of base feature.</td>
    </tr>
    <tr>
      <th>29</th>
      <td>weekday</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of Weekday.</td>
    </tr>
    <tr>
      <th>30</th>
      <td>seconds</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of seconds.</td>
    </tr>
    <tr>
      <th>31</th>
      <td>or</td>
      <td>transform</td>
      <td>For two boolean values, determine if one value is 'True'.</td>
    </tr>
    <tr>
      <th>32</th>
      <td>cum_sum</td>
      <td>transform</td>
      <td>Calculates the sum of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
    <tr>
      <th>33</th>
      <td>latitude</td>
      <td>transform</td>
      <td>Returns the first value of the tuple base feature.</td>
    </tr>
    <tr>
      <th>34</th>
      <td>months</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of months.</td>
    </tr>
    <tr>
      <th>35</th>
      <td>multiply</td>
      <td>transform</td>
      <td>Creates a transform feature that multplies two features.</td>
    </tr>
    <tr>
      <th>36</th>
      <td>subtract</td>
      <td>transform</td>
      <td>Creates a transform feature that subtracts two features.</td>
    </tr>
    <tr>
      <th>37</th>
      <td>weeks</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of weeks.</td>
    </tr>
    <tr>
      <th>38</th>
      <td>days</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of days.</td>
    </tr>
    <tr>
      <th>39</th>
      <td>numwords</td>
      <td>transform</td>
      <td>Returns the words in a given string by counting the spaces.</td>
    </tr>
    <tr>
      <th>40</th>
      <td>percentile</td>
      <td>transform</td>
      <td>For each value of the base feature, determines the percentile in relation</td>
    </tr>
    <tr>
      <th>41</th>
      <td>second</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the second.</td>
    </tr>
    <tr>
      <th>42</th>
      <td>and</td>
      <td>transform</td>
      <td>For two boolean values, determine if both values are 'True'.</td>
    </tr>
    <tr>
      <th>43</th>
      <td>mod</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two features.</td>
    </tr>
    <tr>
      <th>44</th>
      <td>diff</td>
      <td>transform</td>
      <td>Compute the difference between the value of a base feature and the previous value.</td>
    </tr>
    <tr>
      <th>45</th>
      <td>is_null</td>
      <td>transform</td>
      <td>For each value of base feature, return 'True' if value is null.</td>
    </tr>
    <tr>
      <th>46</th>
      <td>add</td>
      <td>transform</td>
      <td>Creates a transform feature that adds two features.</td>
    </tr>
    <tr>
      <th>47</th>
      <td>characters</td>
      <td>transform</td>
      <td>Return the characters in a given string.</td>
    </tr>
    <tr>
      <th>48</th>
      <td>week</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the week.</td>
    </tr>
    <tr>
      <th>49</th>
      <td>cum_count</td>
      <td>transform</td>
      <td>Calculates the number of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
    <tr>
      <th>50</th>
      <td>weekend</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of Weekend.</td>
    </tr>
    <tr>
      <th>51</th>
      <td>hours</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of hours.</td>
    </tr>
    <tr>
      <th>52</th>
      <td>cum_mean</td>
      <td>transform</td>
      <td>Calculates the mean of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
    <tr>
      <th>53</th>
      <td>divide</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two features.</td>
    </tr>
    <tr>
      <th>54</th>
      <td>haversine</td>
      <td>transform</td>
      <td>Calculate the approximate haversine distance in miles between two LatLong variable types.</td>
    </tr>
    <tr>
      <th>55</th>
      <td>year</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the year.</td>
    </tr>
    <tr>
      <th>56</th>
      <td>time_since_previous</td>
      <td>transform</td>
      <td>Compute the time since the previous instance.</td>
    </tr>
    <tr>
      <th>57</th>
      <td>month</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the month.</td>
    </tr>
    <tr>
      <th>58</th>
      <td>isin</td>
      <td>transform</td>
      <td>For each value of the base feature, checks whether it is in a provided list.</td>
    </tr>
    <tr>
      <th>59</th>
      <td>minutes</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number of minutes.</td>
    </tr>
    <tr>
      <th>60</th>
      <td>days_since</td>
      <td>transform</td>
      <td>For each value of the base feature, compute the number of days between it</td>
    </tr>
    <tr>
      <th>61</th>
      <td>cum_max</td>
      <td>transform</td>
      <td>Calculates the max of previous values of an instance for each value in a time-dependent entity.</td>
    </tr>
  </tbody>
</table>
</div>



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>If featuretools does not have enough primitives for us, we can <a href="https://docs.featuretools.com/automated_feature_engineering/primitives.html#defining-custom-primitives">also make our own.</a></p>
<p>To get an idea of what a feature primitive actually does, let's try out a few on our data. Using primitives is surprisingly easy using the <code>ft.dfs</code> function (which stands for deep feature synthesis). In this function, we specify the entityset to use; the <code>target_entity</code>, which is the dataframe we want to make the features for (where the features end up); the <code>agg_primitives</code> which are the aggregation feature primitives; and the <code>trans_primitives</code> which are the transformation primitives to apply.</p>
<p>In the following example, we are using the <code>EntitySet</code> we already created, the target entity is the <code>clients</code> dataframe because we want to make new features about each client, and then we specify a few aggregation and transformation primitives.</p>

</div>
</div>


```python
# Create new features using specified primitives
features, feature_names = ft.dfs(entityset = es, target_entity = 'client', 
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives = ['years', 'month', 'subtract', 'divide'])
```


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>MEAN(loans.loan_amount)</th>
      <th>MEAN(loans.rate)</th>
      <th>MAX(loans.loan_amount)</th>
      <th>MAX(loans.rate)</th>
      <th>LAST(loans.loan_type)</th>
      <th>LAST(loans.loan_amount)</th>
      <th>...</th>
      <th>income / income - log_income</th>
      <th>MAX(loans.loan_amount) / credit_score</th>
      <th>LAST(loans.loan_amount) / join_month - log_income</th>
      <th>MAX(loans.loan_amount) / MEAN(loans.rate)</th>
      <th>log_income - income / LAST(loans.loan_amount)</th>
      <th>join_month / LAST(loans.loan_amount)</th>
      <th>credit_score - join_month / MAX(loans.rate)</th>
      <th>credit_score - income / MEAN(loans.loan_amount)</th>
      <th>credit_score - income / MAX(payment.payment_amount)</th>
      <th>log_income - join_month / credit_score - income</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>3.477000</td>
      <td>13913</td>
      <td>9.44</td>
      <td>home</td>
      <td>2203</td>
      <td>...</td>
      <td>1.000058</td>
      <td>22.404187</td>
      <td>-974.084224</td>
      <td>4001.438021</td>
      <td>-95.964475</td>
      <td>0.004539</td>
      <td>64.724576</td>
      <td>-26.469403</td>
      <td>-77.958950</td>
      <td>-0.000011</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>227920</td>
      <td>633</td>
      <td>5</td>
      <td>12.336750</td>
      <td>7270.062500</td>
      <td>2.517500</td>
      <td>13464</td>
      <td>6.73</td>
      <td>credit</td>
      <td>5275</td>
      <td>...</td>
      <td>1.000054</td>
      <td>21.270142</td>
      <td>-718.983204</td>
      <td>5348.162860</td>
      <td>-43.205244</td>
      <td>0.000948</td>
      <td>93.313522</td>
      <td>-31.263418</td>
      <td>-85.510534</td>
      <td>-0.000032</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>174532</td>
      <td>680</td>
      <td>8</td>
      <td>12.069863</td>
      <td>7824.722222</td>
      <td>2.466111</td>
      <td>14865</td>
      <td>6.51</td>
      <td>other</td>
      <td>13918</td>
      <td>...</td>
      <td>1.000069</td>
      <td>21.860294</td>
      <td>-3419.770809</td>
      <td>6027.708943</td>
      <td>-12.539153</td>
      <td>0.000575</td>
      <td>103.225806</td>
      <td>-22.218297</td>
      <td>-59.294679</td>
      <td>-0.000023</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>214516</td>
      <td>806</td>
      <td>11</td>
      <td>12.276140</td>
      <td>7125.933333</td>
      <td>2.855333</td>
      <td>14593</td>
      <td>5.65</td>
      <td>cash</td>
      <td>9249</td>
      <td>...</td>
      <td>1.000057</td>
      <td>18.105459</td>
      <td>-7247.639641</td>
      <td>5110.786832</td>
      <td>-23.192099</td>
      <td>0.001189</td>
      <td>140.707965</td>
      <td>-29.990457</td>
      <td>-77.207370</td>
      <td>-0.000006</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>38354</td>
      <td>523</td>
      <td>8</td>
      <td>10.554614</td>
      <td>9813.000000</td>
      <td>3.445000</td>
      <td>14837</td>
      <td>6.76</td>
      <td>home</td>
      <td>7223</td>
      <td>...</td>
      <td>1.000275</td>
      <td>28.369025</td>
      <td>-2827.432914</td>
      <td>4306.821480</td>
      <td>-5.308521</td>
      <td>0.001108</td>
      <td>76.183432</td>
      <td>-3.855192</td>
      <td>-13.054175</td>
      <td>-0.000068</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 797 columns</p>
</div>




```python
features.dtypes
```




    income                                                        int64
    credit_score                                                  int64
    join_month                                                    int64
    log_income                                                  float64
    MEAN(loans.loan_amount)                                     float64
    MEAN(loans.rate)                                            float64
    MAX(loans.loan_amount)                                        int64
    MAX(loans.rate)                                             float64
    LAST(loans.loan_type)                                        object
    LAST(loans.loan_amount)                                       int64
    LAST(loans.rate)                                            float64
    LAST(loans.repaid)                                            int64
    MEAN(payment.payment_amount)                                float64
    MAX(payment.payment_amount)                                   int64
    LAST(payment.payment_amount)                                  int64
    LAST(payment.missed)                                          int64
    MONTH(joined)                                                 int64
    credit_score - log_income                                   float64
    join_month - log_income                                     float64
    join_month - credit_score                                     int64
    join_month - income                                           int64
    income - log_income                                         float64
    log_income - credit_score                                   float64
    credit_score - join_month                                     int64
    log_income - join_month                                     float64
    log_income - income                                         float64
    income - join_month                                           int64
    income - credit_score                                         int64
    credit_score - income                                         int64
    join_month / credit_score                                   float64
                                                                 ...   
    MEAN(loans.rate) / income - join_month                      float64
    join_month / log_income - credit_score                      float64
    join_month - credit_score / log_income - join_month         float64
    join_month - credit_score / income - join_month             float64
    income - log_income / income - credit_score                 float64
    income - credit_score / MEAN(payment.payment_amount)        float64
    MEAN(loans.loan_amount) / MAX(payment.payment_amount)       float64
    join_month / income - credit_score                          float64
    join_month - income / LAST(loans.rate)                      float64
    MEAN(payment.payment_amount) / MAX(loans.rate)              float64
    MEAN(loans.loan_amount) / income - credit_score             float64
    MAX(loans.loan_amount) / join_month - income                float64
    credit_score / income - join_month                          float64
    LAST(loans.rate) / MEAN(loans.loan_amount)                  float64
    MEAN(payment.payment_amount) / join_month - credit_score    float64
    log_income / MAX(loans.rate)                                float64
    income - credit_score / log_income - join_month             float64
    income - credit_score / income - join_month                 float64
    MAX(loans.rate) / income - log_income                       float64
    income - credit_score / MAX(loans.loan_amount)              float64
    income / income - log_income                                float64
    MAX(loans.loan_amount) / credit_score                       float64
    LAST(loans.loan_amount) / join_month - log_income           float64
    MAX(loans.loan_amount) / MEAN(loans.rate)                   float64
    log_income - income / LAST(loans.loan_amount)               float64
    join_month / LAST(loans.loan_amount)                        float64
    credit_score - join_month / MAX(loans.rate)                 float64
    credit_score - income / MEAN(loans.loan_amount)             float64
    credit_score - income / MAX(payment.payment_amount)         float64
    log_income - join_month / credit_score - income             float64
    Length: 797, dtype: object




```python
pd.DataFrame(features['MEAN(loans.rate) / join_month - log_income'].head())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MEAN(loans.rate) / join_month - log_income</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>-1.537399</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>-0.343136</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>-0.605944</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>-2.237477</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>-1.348540</td>
    </tr>
  </tbody>
</table>
</div>



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Already we can see how useful featuretools is: it performed the same operations we did manually but also many more in addition. Examining the names of the features in the dataframe brings us to the final piece of the puzzle: deep features.</p>
<h2>Deep Feature Synthesis<a class="anchor-link" href="#Deep-Feature-Synthesis">¶</a>
</h2>
<p>While feature primitives are useful by themselves, the main benefit of using featuretools arises when we stack primitives to get deep features. The depth of a feature is simply the number of primitives required to make a feature. So, a feature that relies on a single aggregation would be a deep feature with a depth of 1, a feature that stacks two primitives would have a depth of 2 and so on. The idea itself is lot simpler than the name "deep feature synthesis" implies. (I think the authors were trying to ride the way of deep neural network hype when they named the method!) To read more about deep feature synthesis, check out <a href="https://docs.featuretools.com/automated_feature_engineering/afe.html">the documentation</a> or the <a href="http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf">original paper by Max Kanter et al</a>.</p>
<p>Already in the dataframe we made by specifying the primitives manually we can see the idea of feature depth. For instance, the MEAN(loans.loan_amount) feature has a depth of 1 because it is made by applying a single aggregation primitive. This feature represents the average size of a client's previous loans.</p>

</div>
</div>


```python
# Create new features using specified primitives
features, feature_names = ft.dfs(entityset = es, target_entity = 'client', 
                                 agg_primitives = ['mean'],
                                 trans_primitives = ['subtract', 'divide'])
```


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>credit_score</th>
      <th>join_month</th>
      <th>log_income</th>
      <th>MEAN(loans.loan_amount)</th>
      <th>MEAN(loans.rate)</th>
      <th>MEAN(payment.payment_amount)</th>
      <th>credit_score - log_income</th>
      <th>log_income - join_month</th>
      <th>log_income - income</th>
      <th>...</th>
      <th>income - credit_score / MEAN(payment.payment_amount)</th>
      <th>join_month / income - credit_score</th>
      <th>MEAN(loans.loan_amount) / income - credit_score</th>
      <th>credit_score / income - join_month</th>
      <th>MEAN(payment.payment_amount) / join_month - credit_score</th>
      <th>income - credit_score / log_income - join_month</th>
      <th>income - credit_score / income - join_month</th>
      <th>income / income - log_income</th>
      <th>credit_score - income / MEAN(loans.loan_amount)</th>
      <th>log_income - join_month / credit_score - income</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>211422</td>
      <td>621</td>
      <td>10</td>
      <td>12.261611</td>
      <td>7963.950000</td>
      <td>3.477000</td>
      <td>1178.552795</td>
      <td>608.738389</td>
      <td>2.261611</td>
      <td>-211409.738389</td>
      <td>...</td>
      <td>178.864282</td>
      <td>0.000047</td>
      <td>0.037779</td>
      <td>0.002937</td>
      <td>-1.928892</td>
      <td>93208.319781</td>
      <td>0.997110</td>
      <td>1.000058</td>
      <td>-26.469403</td>
      <td>-0.000011</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>227920</td>
      <td>633</td>
      <td>5</td>
      <td>12.336750</td>
      <td>7270.062500</td>
      <td>2.517500</td>
      <td>1166.736842</td>
      <td>620.663250</td>
      <td>7.336750</td>
      <td>-227907.663250</td>
      <td>...</td>
      <td>194.805711</td>
      <td>0.000022</td>
      <td>0.031986</td>
      <td>0.002777</td>
      <td>-1.857861</td>
      <td>30979.248435</td>
      <td>0.997245</td>
      <td>1.000054</td>
      <td>-31.263418</td>
      <td>-0.000032</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>174532</td>
      <td>680</td>
      <td>8</td>
      <td>12.069863</td>
      <td>7824.722222</td>
      <td>2.466111</td>
      <td>1207.433824</td>
      <td>667.930137</td>
      <td>4.069863</td>
      <td>-174519.930137</td>
      <td>...</td>
      <td>143.984703</td>
      <td>0.000046</td>
      <td>0.045008</td>
      <td>0.003896</td>
      <td>-1.796777</td>
      <td>42716.912967</td>
      <td>0.996150</td>
      <td>1.000069</td>
      <td>-22.218297</td>
      <td>-0.000023</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>214516</td>
      <td>806</td>
      <td>11</td>
      <td>12.276140</td>
      <td>7125.933333</td>
      <td>2.855333</td>
      <td>1109.473214</td>
      <td>793.723860</td>
      <td>1.276140</td>
      <td>-214503.723860</td>
      <td>...</td>
      <td>192.622947</td>
      <td>0.000051</td>
      <td>0.033344</td>
      <td>0.003757</td>
      <td>-1.395564</td>
      <td>167466.003631</td>
      <td>0.996294</td>
      <td>1.000057</td>
      <td>-29.990457</td>
      <td>-0.000006</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>38354</td>
      <td>523</td>
      <td>8</td>
      <td>10.554614</td>
      <td>9813.000000</td>
      <td>3.445000</td>
      <td>1439.433333</td>
      <td>512.445386</td>
      <td>2.554614</td>
      <td>-38343.445386</td>
      <td>...</td>
      <td>26.281870</td>
      <td>0.000211</td>
      <td>0.259390</td>
      <td>0.013639</td>
      <td>-2.795016</td>
      <td>14808.890291</td>
      <td>0.986570</td>
      <td>1.000275</td>
      <td>-3.855192</td>
      <td>-0.000068</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 396 columns</p>
</div>




```python
# Show a feature with a depth of 1
pd.DataFrame(features['MEAN(loans.loan_amount)'].head(10))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MEAN(loans.loan_amount)</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>7963.950000</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>7270.062500</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>7824.722222</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>7125.933333</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>9813.000000</td>
    </tr>
    <tr>
      <th>32726</th>
      <td>6633.263158</td>
    </tr>
    <tr>
      <th>32885</th>
      <td>9920.400000</td>
    </tr>
    <tr>
      <th>32961</th>
      <td>7882.235294</td>
    </tr>
    <tr>
      <th>35089</th>
      <td>6939.200000</td>
    </tr>
    <tr>
      <th>35214</th>
      <td>7173.555556</td>
    </tr>
  </tbody>
</table>
</div>



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As well scroll through the features, we see a number of features with a depth of 2. For example, the LAST(loans.(MEAN(payments.payment_amount))) has depth = 2 because it is made by stacking two feature primitives, first an aggregation and then a transformation. This feature represents the average payment amount for the last (most recent) loan for each client.</p>

</div>
</div>

<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can create features of arbitrary depth by stacking more primitives. However, when I have used featuretools I've never gone beyond a depth of 2! After this point, the features become very convoluted to understand. I'd encourage anyone interested to experiment with increasing the depth (maybe for a real problem) and see if there is value to "going deeper".</p>
<h2>Automated Deep Feature Synthesis<a class="anchor-link" href="#Automated-Deep-Feature-Synthesis">¶</a>
</h2>
<p>In addition to manually specifying aggregation and transformation feature primitives, we can let featuretools automatically generate many new features. We do this by making the same <code>ft.dfs</code> function call, but without passing in any primitives. We just set the <code>max_depth</code> parameter and featuretools will automatically try many all combinations of feature primitives to the ordered depth.</p>
<p>When running on large datasets, this process can take quite a while, but for our example data, it will be relatively quick. For this call, we only need to specify the <code>entityset</code>, the <code>target_entity</code> (which will again be <code>clients</code>), and the <code>max_depth</code>.</p>

</div>
</div>


```python
# Perform deep feature synthesis without specifying primitives
features, feature_names = ft.dfs(entityset=es, target_entity='client', 
                                 max_depth = 2)
```


```python
features.iloc[:, 4:].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SUM(loans.loan_amount)</th>
      <th>SUM(loans.rate)</th>
      <th>STD(loans.loan_amount)</th>
      <th>STD(loans.rate)</th>
      <th>MAX(loans.loan_amount)</th>
      <th>MAX(loans.rate)</th>
      <th>SKEW(loans.loan_amount)</th>
      <th>SKEW(loans.rate)</th>
      <th>MIN(loans.loan_amount)</th>
      <th>MIN(loans.rate)</th>
      <th>...</th>
      <th>NUM_UNIQUE(loans.WEEKDAY(loan_end))</th>
      <th>MODE(loans.MODE(payment.missed))</th>
      <th>MODE(loans.DAY(loan_start))</th>
      <th>MODE(loans.DAY(loan_end))</th>
      <th>MODE(loans.YEAR(loan_start))</th>
      <th>MODE(loans.YEAR(loan_end))</th>
      <th>MODE(loans.MONTH(loan_start))</th>
      <th>MODE(loans.MONTH(loan_end))</th>
      <th>MODE(loans.WEEKDAY(loan_start))</th>
      <th>MODE(loans.WEEKDAY(loan_end))</th>
    </tr>
    <tr>
      <th>client_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25707</th>
      <td>159279</td>
      <td>69.54</td>
      <td>4044.418728</td>
      <td>2.421285</td>
      <td>13913</td>
      <td>9.44</td>
      <td>-0.172074</td>
      <td>0.679118</td>
      <td>1212</td>
      <td>0.33</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>2010</td>
      <td>2007</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26326</th>
      <td>116321</td>
      <td>40.28</td>
      <td>4254.149422</td>
      <td>1.991819</td>
      <td>13464</td>
      <td>6.73</td>
      <td>0.135246</td>
      <td>1.067853</td>
      <td>1164</td>
      <td>0.50</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>2003</td>
      <td>2005</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26695</th>
      <td>140845</td>
      <td>44.39</td>
      <td>4078.228493</td>
      <td>1.517660</td>
      <td>14865</td>
      <td>6.51</td>
      <td>0.154467</td>
      <td>0.820060</td>
      <td>2389</td>
      <td>0.22</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>2003</td>
      <td>2005</td>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26945</th>
      <td>106889</td>
      <td>42.83</td>
      <td>4389.555657</td>
      <td>1.564795</td>
      <td>14593</td>
      <td>5.65</td>
      <td>0.156534</td>
      <td>-0.001998</td>
      <td>653</td>
      <td>0.13</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>2002</td>
      <td>2004</td>
      <td>12</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29841</th>
      <td>176634</td>
      <td>62.01</td>
      <td>4090.630609</td>
      <td>2.063092</td>
      <td>14837</td>
      <td>6.76</td>
      <td>-0.212397</td>
      <td>0.050600</td>
      <td>2778</td>
      <td>0.26</td>
      <td>...</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
      <td>2005</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Deep feature synthesis has created 90 new features out of the existing data! While we could have created all of these manually, I am glad to not have to write all that code by hand. The primary benefit of featuretools is that it creates features without any subjective human biases. Even a human with considerable domain knowledge will be limited by their imagination when making new features (not to mention time). Automated feature engineering is not limited by these factors (instead it's limited by computation time) and provides a good starting point for feature creation. This process likely will not remove the human contribution to feature engineering completely because a human can still use domain knowledge and machine learning expertise to select the most important features or build new features from those suggested by automated deep feature synthesis.</p>
<h1>Next Steps<a class="anchor-link" href="#Next-Steps">¶</a>
</h1>
<p>While automatic feature engineering solves one problem, it provides us with another problem: too many features! Although it's difficult to say which features will be important to a given machine learning task ahead of time, it's likely that not all of the features made by featuretools add value. In fact, having too many features is a significant issue in machine learning because it makes training a model much harder. The <a href="https://pdfs.semanticscholar.org/a83b/ddb34618cc68f1014ca12eef7f537825d104.pdf">irrelevant features can drown out the important features</a>, leaving a model unable to learn how to map the features to the target.</p>
<p>This problem is known as the <a href="https://en.wikipedia.org/wiki/Curse_of_dimensionality#Machine_learning">"curse of dimensionality"</a> and is addressed through the process of <a href="http://scikit-learn.org/stable/modules/feature_selection.html">feature reduction and selection</a>, which means <a href="https://machinelearningmastery.com/feature-selection-machine-learning-python/">removing low-value features</a> from the data. Defining which features are useful is an important problem where a data scientist can still add considerable value to the feature engineering task. Feature reduction will have to be another topic for another day!</p>

</div>
</div>

<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1>Conclusions<a class="anchor-link" href="#Conclusions">¶</a>
</h1>
<p>In this notebook, we saw how to apply automated feature engineering to an example dataset. This is a powerful method which allows us to overcome the human limits of time and imagination to create many new features from multiple tables of data. Featuretools is built on the idea of deep feature synthesis, which means stacking multiple simple feature primitives - <strong>aggregations and transformations</strong> - to create new features. Feature engineering allows us to combine information across many tables into a single dataframe that we can then use for machine learning model training. Finally, the next step after creating all of these features is figuring out which ones are important.</p>
<p>Featuretools is currently the only Python option for this process, but with the recent emphasis on automating aspects of the machine learning pipeline, other competitiors will probably enter the sphere. While the exact tools will change, the idea of automatically creating new features out of existing data will grow in importance. Staying up-to-date on methods such as automated feature engineering is crucial in the rapidly changing field of data science. Now go out there and find a problem on which to apply featuretools!</p>
<p>For more information, check out the <a href="https://docs.featuretools.com/index.html">documentation for featuretools</a>. Also, read about how featuretools is <a href="https://www.featurelabs.com/">used in the real world by Feature Labs</a>, the company behind the open-source library.</p>

</div>
</div>
