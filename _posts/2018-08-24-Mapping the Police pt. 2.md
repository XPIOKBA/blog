---
layout: post
title: Mapping the Police, pt .2
summary: Police department sizes and population in Allegheny County, PA
---
_[Git](https://github.com/afriedman412/police_mapping_ag_county)_   
This is the second entry in an ongoing series of posts on the scope of policing in the USA. [The first entry](https://afriedman412.github.io/Mapping-the-Police-v1.0/) explained my motivation for this research, and made pretty basic national survey of America's police departments using data scraped from the website [PoliceOne](http://www.policeone.com).

The data on PoliceOne is self-reported, incomplete, and unverified, and limited the utility of the first pass. I knew for whatever next step I took, I would have to dig for more accurate information. So I decided to limit my search to Allegheny County, Pennsylvania, which includes Pittsburgh.

# Gathering the data
---
The shapefiles for the municipalities in Allegheny County were all available from the [Western Pennsylvania Regional Data Center](https://data.wprdc.org/dataset/allegheny-county-municipal-boundaries). 2017 population estimates are from the [Census Bureau](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml).

The [Allegheny County website](http://www.alleghenycounty.us/emergency-services/police-departments.aspx) has a decent directory of its police departments and their jurisdictions. However, information on the size of these departments was harder to come by, let alone data from the police themselves. Some listed their staff on local websites, although the design of the sites raises questions about how recent that information might be. I used newspaper articles and PoliceOne to fill in what was missing. 

These numbers are still incomplete (missing about 8 of the 130 departments listed), and I wouldn't vouch for the total accuracy of the officer counts. But I'm confident they are reasonable ballpark estimates.

# Data exploration
---
We start with a look at population distribution across the county. Not surprisingly, the city of Pittsburgh is by far the largest municipality (population 300k, almost ten times larger than Penn Hills, at a population of roughly 41k). This map uses the log of the 2017 population estimate to allow for better visualization.

![png](images/technical%20notebook_files/technical%20notebook_23_0.png)

A geographical plot of 'log officers per 1k' gives a qualitative look at levels of policing compare across the county. Again, using 'log' instead of the raw numbers to better visualize a data set with some heavy outliers.

![png](images/technical%20notebook_files/technical%20notebook_27_0.png)

This plot shows the relationship between the population and the size of a police force. Not surprisingly, as one grows, so does the other. 

![png](images/technical%20notebook_files/technical%20notebook_29_0.png)

It's a little more interesting look at the relationship between the population and the ratio of the number of officers in a department to the number of people in its jurisdiction. Here we see the log of the 2017 population estimate ('log pop 2017') and the log of the ratio of officers to 1000 people ('log officers per 1k'). As the population grows, the relative size of its police force shrinks.

![png](images/technical%20notebook_files/technical%20notebook_31_0.png)

It's easy to come up with guesses as to why this is the case. For example, a larger population is likely to be more dense, allowing police to work more efficiently. Also, this ratio is increasingly unstable for smaller departments: a difference of one officer has a much larger impact on a 5-person department than one with 20 or more people. (This is especially worth noting given the questions about the accuracy of the department size data).

Given a reasonably predictable relationship between population and the ratio of officers to population, we can measure whether a given municipality has more or less police than expected by how far the actual ratio deviates from the expected ratio.

We visualize this with a residual plot.

![png](images/technical%20notebook_files/technical%20notebook_33_0.png)

By running a simple linear regression, we can more effectively dig into this data. The 'residual' column is the measured deviation, while 'abs residual' is the absolute value of the deviation.

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>agency</th>
      <th>pop 2017</th>
      <th>residual</th>
      <th>abs residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128</th>
      <td>Pittsburgh P.D.</td>
      <td>302407</td>
      <td>1.314907</td>
      <td>1.314907</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Glassport P.D.</td>
      <td>4367</td>
      <td>-0.669603</td>
      <td>0.669603</td>
    </tr>
    <tr>
      <th>71</th>
      <td>McKeesport P.D.</td>
      <td>19245</td>
      <td>0.531176</td>
      <td>0.531176</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Springdale Township P.D.</td>
      <td>1596</td>
      <td>0.468108</td>
      <td>0.468108</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Brackenridge P.D.</td>
      <td>3179</td>
      <td>-0.404315</td>
      <td>0.404315</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Scott Township P.D.</td>
      <td>16623</td>
      <td>-0.375358</td>
      <td>0.375358</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fawn Township P.D.</td>
      <td>2333</td>
      <td>-0.361984</td>
      <td>0.361984</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Rosslyn Farms P.D.</td>
      <td>415</td>
      <td>-0.347612</td>
      <td>0.347612</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Wilkinsburg P.D.</td>
      <td>15554</td>
      <td>0.342251</td>
      <td>0.342251</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Ingram P.D.</td>
      <td>3248</td>
      <td>-0.328072</td>
      <td>0.328072</td>
    </tr>
  </tbody>
</table>
</div>

The city of Pittsburgh differs from its expected ratio the most by far, which raises questions about the range of populations in which we can expect the observed trends to apply. 

Since we are doing all this analysis with a larger eye towards the effects of department consolidation, how do the agencies which cover more than one jurisdiction rank?

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>agency</th>
      <th>pop 2017</th>
      <th>jurisdictions</th>
      <th>residual</th>
      <th>abs residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Northern Regional P.D.</td>
      <td>34800</td>
      <td>4</td>
      <td>-0.254589</td>
      <td>0.254589</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crafton P.D.</td>
      <td>6246</td>
      <td>2</td>
      <td>-0.241441</td>
      <td>0.241441</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ohio Township P.D.</td>
      <td>15459</td>
      <td>8</td>
      <td>0.211811</td>
      <td>0.211811</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sewickley P.D.</td>
      <td>4381</td>
      <td>2</td>
      <td>0.153868</td>
      <td>0.153868</td>
    </tr>
    <tr>
      <th>8</th>
      <td>White Oak P.D.</td>
      <td>8014</td>
      <td>2</td>
      <td>-0.150602</td>
      <td>0.150602</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North Versailles P.D.</td>
      <td>12161</td>
      <td>2</td>
      <td>0.144524</td>
      <td>0.144524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elizabeth Borough P.D.</td>
      <td>1984</td>
      <td>2</td>
      <td>-0.135696</td>
      <td>0.135696</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Carnegie P.D.</td>
      <td>8559</td>
      <td>2</td>
      <td>-0.124841</td>
      <td>0.124841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East McKeesport P.D.</td>
      <td>2646</td>
      <td>2</td>
      <td>-0.078178</td>
      <td>0.078178</td>
    </tr>
  </tbody>
</table>
</div>

Compared to the largest deviations in the first chart of over 0.32, these values follow established trends.

## Conclusion and next ste
---
There is not much to be concluded from this basic analysis of population and policing in and of itself. It's impossible to make reasonable statements about how many police any area needs without considering data on criminal activity in the area or how much of the municipal budget goes to public safety.

For example: Wilkinsburg is ostensibly "overpoliced" in terms of how many officers we would expect it to have given its population. As an anecdotally high crime area, it might make sense to have a larger police force (if we ignore, for the moment, questions about whether more police actually reduce crime). This, in turn, is complicated by its budget problems. A high crime area may have a need for more police, but a more pressing need to spend its money elsewhere.

Adding this data into the mix is an obvious next step, and all these observations are starting points for further analysis. In the mean time, this rough look at police/population ratios is a good baseline for doing the same research in other metropolitan areas. I will probably start with St. Louis and Baltimore.
