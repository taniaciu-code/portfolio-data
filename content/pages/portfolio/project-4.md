---
title: The Effect of Average Temperature & Rainfall on Flood in Indonesia
subtitle: Aug 2020 - Dec 2020
date: '2019-02-26'
thumb_image: /images/flood.jpg
thumb_image_alt: A table tennis racket on a pink background
image: /images/flood-d4fb0e55.jpg
image_alt: A table tennis racket on a pink background
seo:
  title: Project Title 4
  description: This is the project 4 description
  extra:
    - name: 'og:type'
      value: website
      keyName: property
    - name: 'og:title'
      value: Project Title 4
      keyName: property
    - name: 'og:description'
      value: This is the project 4 description
      keyName: property
    - name: 'og:image'
      value: images/4.jpg
      keyName: property
      relativeUrl: true
    - name: 'twitter:card'
      value: summary_large_image
    - name: 'twitter:title'
      value: Project Title 4
    - name: 'twitter:description'
      value: This is the project 4 description
    - name: 'twitter:image'
      value: images/4.jpg
      relativeUrl: true
layout: project
---
<div align="justify">
The impact of flooding which is quite extensive on the community makes flooding becomes one of the most detrimental disasters in Indonesia. It cannot be denied that flooding has become a government concern in the last decade. There are a number of factors that can influence the occurrence of flooding such as temperature and precipitation levels. However, not many studies have discussed the effect of average temperature and precipitation on flooding. For this reason, this study was conducted to find out the effect of average temperature and precipitation on flooding in Indonesia. In this study, flood data, climate data, and geospatial datasets were used. These data will be visualized by using Tableau workbooks. Data visualization is done by combining the data with the left join technique. After that, data will be visualized by using the concept of level of detail, mapping, clustering, trend analysis, and forecasting as well as a number of charts such as line charts, bar charts, pareto charts, control diagrams, and others. With this research, it is hoped that it can add insight and knowledge to readers regarding the effect of average temperature and rainfall levels on floods in Indonesia.
</div>

## Exploration Visualization

### Dashboard | Average Temperature with 3 LOD Type

![png](/images/Das1.png)
<div align="justify">
The above dashboard comes with a bar chart showing the differences between the three LOD expressions. Exclude is orange, Fixed is red, and Include is blue. The purpose of the bar chart is to make it easier to compare the three different LOD values. The Text Table on the dashboard above is also used as a filter, where when a row in the text table is clicked, the bar chart will change according to the row that was clicked.
</div>

### Dashboard | Flood in Indonesia

![png](/images/Das2.png)

<div align="justify">
In the dashboard above, the researcher uses the tools available in Tableau to visualize flood disaster data in Indonesia. The map made consists of a map of Indonesia, which is divided by district/city using data from GADM. On the map, tooltips are equipped with the name of the selected district/city, the number of floods in the 1985-2020 period, and a bar chart showing how many victims died due to flooding in the area.
<br>
At the top right of the dashboard, you can see a bar chart showing the number of floods per year in the districts/cities selected by the user on the map on the left. Meanwhile, at the bottom right of the dashboard, there is a web page object that can access Wikipedia. The web page object will open the Wikipedia page that matches the user's choice on the map.
</div>

### Dashboard | Flood Frequency and Average Temperature

![png](/images/Das3.png)

<div align="justify">
In the dashboard above, it can be seen that there is a control diagram that represents the development of the average temperature in Indonesia and a Pareto chart that represents the duration of flooding by province in Indonesia. Control diagram of the average temperature is used to monitor the development of the average temperature over time under controlled conditions. Based on the control diagram, it is found that the upper control limit of the average temperature in Indonesia is 27.3475oC, while the lower control limit of the average temperature in Indonesia is 26.9496oC. The results of the control diagram show that from 1985 to 2020 it was found that in 1987, 1998, and 2016 Indonesia had an average temperature that exceeded the upper control limit with the highest average temperature of 27.5309oC in 1987.
<br>
In addition, the results of the control diagram also show that from 1985 to 2020 it was found that in 1991, 1999, 2008, and 2011 Indonesia had an average temperature below the lower control limit with the lowest average temperature of 24,158oC in 2001. Thus, it can also be obtained that from 1985 to 2020 Indonesia had a fairly controlled average temperature. This can be seen from the control diagram which shows that most of Indonesia's average temperature is in the upper control limit and lower control limit ranges.
<br>
The Pareto chart in this study is used to see the frequency of floods by province. Pareto chart is basically a dual combination chart which consists of a bar chart which represents the quantity of a categorical variable and a line chart which represents the cumulative total of the quantity of categorical variables. In this study, the bar chart in the Pareto chart represents the number or frequency of floods from categorical variables in the form of the name of the province, while the line chart represents the cumulative total of the number or frequency of floods based on the name of the province.
</div>

## Prediction Visualization

### Dashboard | Region and Flood Duration Clustering in Indonesia

![png](/images/Das4.png)

<div align="justify">
In the dashboard above, it can be seen that there is a scatter plot with the concept of clustering which represents the grouping of regions in Indonesia based on the level of rainfall (precipitation) and average temperature. In addition, the dashboard also contains a bar chart of the length of flooding per month in Indonesia. This dashboard was created with the aim of seeing the effect of rainfall levels and average temperature on the duration of flooding in Indonesia. Because the longer the flood occurs, the greater the losses borne by the community. With the results of the analysis of this visualization, it can be obtained information on the duration of the flood, the level of rainfall, and the average temperature which is used as a basis for taking flood preventive measures or flood disaster mitigation measures in Indonesia.
</div>

### Dashboard | Comparison of 5 Types of Trend Lines between Average Temperature and Rainfall Rates

![png](/images/Das5-1f9e4bef.png)

<div align="justify">
On the dashboard above, there are 5 trend lines, namely Linear, Logarithmic, Exponential, Polynomial, and Power. The five trend lines have a p-Value of more than 0.05, which means that there is no significant trend line at an alpha of 0.05. However, of the 5 trend lines, the lowest p-value is the polynomial trend line.
<br>
Of the 5 trend lines on the dashboard, the polynomial trend line has the highest R-Squared compared to other trend lines, which is around 0.0476. This means that the polynomial trend line can explain about 4% of the variance in the data, whereas when viewed from the standard error, both the exponential trend line and the power trend line have very similar standard error values, which are around 0.059. The exponential trend line has a slightly smaller standard error, and has the smallest standard error compared to other trend lines. Based on the p-Value, R-Squared, and standard error values, it is concluded that the polynomial trend line is the most suitable compared to other trend lines because it has the smallest p-Value and the largest R-Squared among the 5 trend lines on the dashboard.
</div>

### Dashboard | Monthly Forecasting of Average Temperature in Indonesia

![png](/images/Das6.png)

<div align="justify">
The dashboard above consists of a map that displays the location of weather stations in Indonesia, and a line graph that displays the average temperature in Indonesia per month along with its forecast. The forecast results show that there are 3 "peaks" in November 2020, May 2021, and November 2021. Meanwhile, there are 2 "valleys" in February 2021 and July 2021. This pattern is in accordance with the average temperature data in the previous months. From the trend line, it can be seen that in the long term the average temperature in Indonesia has increased from year to year.
</div>

### Dashboard | Quarterly Forecasting of Average Temperature, Rainfall, and Number of Floods in Indonesia

![png](/images/Das7.png)

<div align="justify">
On the dashboard above, you can see a bar chart and 2 line graphs along with their respective forecasts. The bar chart at the top shows the number of floods in Indonesia per quarter, while the two line charts below represent the average rainfall and temperature levels in Indonesia per quarter. When the average temperature forecast increases, the rainfall rate also increases. On the other hand, when the average temperature forecast decreases, the level of rainfall also tends to decrease. This is consistent with the fact that when the average temperature on the earth's surface increases, it will result in more evaporation, which in turn will increase overall rainfall.
</div>