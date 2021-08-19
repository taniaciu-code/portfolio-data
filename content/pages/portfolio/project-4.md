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

## Prediction Visualization

### Dashboard | Region and Flood Duration Clustering in Indonesia

![png](/images/Das4.png)

### Dashboard | Comparison of 5 Types of Trend Lines between Average Temperature and Rainfall Rates

![](/images/Das5-1f9e4bef.png)

### Dashboard | Monthly Forecasting of Average Temperature in Indonesia

![](/images/Das6.png)

### Dashboard | Quarterly Forecasting of Average Temperature, Rainfall, and Number of Floods in Indonesia

![](/images/Das7.png)

