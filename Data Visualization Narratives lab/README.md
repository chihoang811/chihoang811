## DATA VISUALIZATION NARRATIVES LAB (University of Milan)
###### _by Linh Chi Hoang, Ikram Ait Taleb Naser, Viola Awor_
### Project introduction
   - Instructor: Professor Beatrice Gobbo.
   - Main topic: Caro Affitti (Expensive Rents).
   - Data:
     - The data is provided by Professor Gobbo, which is an Excel file named as `"Original data"`.
       - It is obtained from Immobiliare.it.
     - Students can also use data from other sources.
   - Objectives:
     - As a result, students have to make a short speech with their visualizations to present and describe the data.
   - Requirement:
     - The visualizations must be presented in the format of Instagram posts, reels, stories.
     - Students may present on a variety of topics, but they must demonstrate relevance to the main topic, which is "Expensive rents".
### The implementation process
   - Research data:
     - The data used is provided by the Professor.
     - The data contains information about various types of rooms for rent in four big city in Italy namely Milan, Rome, Florence, and Bolgona along with their rent price, addresses, latitude, longitude.
   - Create the Research question:
     - Being aware of the latitude and longitude as well as the rent price of each room, our group has come up with the question "Is it true to say that the room' location determines their rent price?".
     - Our research area is the city of Milan.
   - Sort the data:
     - Because our group focuses only on Milan, we have to sort the data. In order to do this, we use Python to obtain the data only about the city of Milan. As a result, we have an Excel file named `"Sorted data"`.
     - Next, considering the city centre as the based location (Duomo di Milano), we use the longitude and latitude to calculate the distance of each room with the city centre.
     - Lastly, we classify the rooms according to the distances, in which there are 7 groups:
       - 300m to 2km
       - 2km to 3.7km
       - 3.7km to 5.5km
       - 5.5km to 7.2km
       - 7.2km tp 9km
       - 9km to 10.6km
       - 10.6km to 12.3km
   - Visualize the data:
     - In order to create the visualization,
       - First, we use an app named "Flourish", which allows us to easily create beautiful data visualization. For our work, Slope chart and Dot visualization are two types of charts chosen to show the relationship between the rooms' distances and their rent price.
       - Second, we use the app named "Canva" to add design adjustment on the charts.
     - Ultimately, our visualization is finished.
       - In order to see the result, please find these two files in the "Project" folder:
         - `"dataviz lab pic"`
         - `"dataviz lab vid"`
### Conclusions
   - In the beginning, we thought that with the rooms which are far from the city centre, their rent prices are less expensive compared to those around the city centres.
   - However, in contrast to prior belief, the price rent in Milan city is not affected by whether the rooms are close to or far from the city centre.
   - According to the research of other groups, the prices can be varied by other factors such as the availability of metro stops, size of the rooms, easy accessibility to public amenities, etc. 
### Improvement
   - In order to make the work become more applicable, we should consider also the size of the rooms, instead of assuming every room shares the same size.