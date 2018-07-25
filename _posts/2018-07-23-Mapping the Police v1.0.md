---
layout: post
title: Mapping the Police v. 1.0
summary: A first draft at mapping America's police departments.
---
_[Git](https://github.com/afriedman412/mapping_the_police)_  
In June of 2018, East Pittsburgh Police officer Michael Rosfeld shot 17 year old Antwan Rose Jr. in the back, killing him. Rose was one of several passengers fleeing a jitney which Police had stopped for being visually connected to a recent shooting. But Rose did not know the other passengers and had not committed any crime. In a surprising turn of events, after heavy protests from the community shut down major roads in Pittsburgh, Officer Rosfeld was arrested and charged with criminal homicide (although he was grainted bail and set free).

Officer Rosfeld's career in law enforcement up to this point was not exactly exemplary. Prior to joining East Pittsburgh PD (East Pittsburgh is a small borough outside the city, not a part of Pittsburgh proper), Rosfeld spent five years on the police force of the University of Pittsburgh. He left after a lawsuit alleged abuses in December of 2017. EPPD hired him a scant four months later.

A side note: In the press conference announcing Rosfeld's indictment, a reporter asked Allegheny County District Attorney Stephen Zappala about EPPD's policies regarding such situations. Zappala responded that, by all accounts, the department [has no policies about anything](https://twitter.com/meganguzaTrib/status/1011996036397780993).

Rosfeld's journey to East Pittsburgh via questionable departure from his last gig made me wonder if the sheer number of police departments in America could contribute to systematic abuse. (Similar questions about the balkanization of suburban St. Louis police came up in the aftermath of Ferguson as well.) So I decided to find a way to map America's police departments.

<br>

# Gathering the data
---
I pulled my data from [PoliceOne](http://www.policeone.com), a police-focused site that features "a comprehensive list of federal, state, and local law enforcement agencies in the United States". Every department on the site has a page with contact information, population served and number of officers. As far as I can tell, all the data is self-reported, or at the very least, not apparently vetted by any official body.

I scraped every department page for its population and number of officers using BeautifulSoup. I also pulled 2016 state population data from the US census for comparison.

As this graph shows, we are missing a lot of people.


![png](../images/mapping_the_police_files/mapping_the_police_18_0.png)


Here's the percentage of the census population covered by the PoliceOne departments by state.


![png](../images/mapping_the_police_files/mapping_the_police_19_0.png)

<br>

# Loading coordinates for mapping
---
I used [GeoPy](https://geopy.readthedocs.io/en/stable/) to find approximate longitude and latitude for every police department. GeoPy can pull coordinates from search terms, so I fed it every PD's name and state with common words like 'Police' and 'Department' stripped out.

<br>

# Mapping the departments
---
I mapped the locations of the departments by using the longitude and latitude as x and y coordinates respectively in matplotlib.

It's clear from this first plot that the GeoPy search wasn't perfect, although the America-shaped blob is encouraging.


![png](../images/mapping_the_police_files/mapping_the_police_31_0.png)


I subset the data on the rough boundaries of the lower 48 from a google search and things look even better.


![png](../images/mapping_the_police_files/mapping_the_police_33_0.png)


And finally, I set the size and color of each point to a scale set by log2 of the number of officers.


![png](../images/mapping_the_police_files/mapping_the_police_35_0.png)

<br>

# Conclusion and next steps
---
As stated at the beginning of the article, this is only a starting point for further analysis. The data is spotty and the map doesn't show us much more than a vague notion that the police are where the people are. However, hopefully I will be able to build this model out further with better data and more advanced mapping options.