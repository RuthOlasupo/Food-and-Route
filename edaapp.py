import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sklearn
import numpy as np
import streamlit.components.v1 as components

# Load the dataset with a specified encoding
data = pd.read_csv('cleaned_data_2024.csv', encoding='latin1')


# Page 1: Dashboard
def dashboard():
    import base64  # Import inside the function to keep scope clean
    
    st.title("Food Drive Prediction")

    # Encode the image as base64
    def get_image_as_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # Convert the image to base64
    image_base64 = get_image_as_base64("logo.png")

    # Use HTML to render the image
    st.markdown(
        f"""
        <div style="text-align: left;">
            <img src="data:image/png;base64,{image_base64}" alt="Logo" style="width: 150px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    

    
    # Additional dashboard components (if needed)
    st.subheader(" 🤝 Welcome to the Food Drive Prediction Dashboard!")
    
    st.write("Use the menu on the left to navigate through the app.")
    
    st.subheader("💡 Brief:")

    inspiration = '''
    The Edmonton Food Drive, organized by the Church of Jesus Christ of Latter Day Saints seeks to bring the community together in a shared effort to care for those in need. Through the generosity of donations and the compassion of volunteers, they aim to provide essential food to families facing difficult times.
    Through this project we discovered how technology tools can be utilised to improve decision making processes and add value to lifes'''
    

    st.write(inspiration)

    st.subheader("👨🏻‍💻 Scope of the Project")

    what_it_does = '''
    The Edmonton Food Drive project initiative leverages machine learning to streamline the management of food donations in Edmonton Alberta. Its goal is to enhance the efficiency of drop off and pick up operations and optimize resource allocation leading to a more effective and impactful food drive campaign.
    '''

    st.write(what_it_does)



# Page 2: Exploratory Data Analysis (EDA) 
def exploratory_data_analysis():
    import base64  # Import inside the function to keep scope clean
    # Set the page title
    st.subheader("Visualization of trends from 2023 to 2024")
    

    # Streamlit App Title
    st.title("Edmonton Food Drive - Time Series Analysis")

    # Embed Tableau Visualization
    tableau_html = """
    <div class='tableauPlaceholder' id='viz1733872310940' style='position: relative'>
      <noscript>
    <a href='#'>
      <img alt='Edmonton Food Drive - Time Series Analysis (2023 - 2024) '
           src='https://public.tableau.com/static/images/ED/EDA-Demo/NoOutliersDashboard/1_rss.png' style='border: none' />
    </a>
      </noscript>
      <object class='tableauViz'  style='display:none;'>
    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
    <param name='embed_code_version' value='3' />
    <param name='site_root' value='' />
    <param name='name' value='EDA-Demo/NoOutliersDashboard' />
    <param name='tabs' value='no' />
    <param name='toolbar' value='yes' />
    <param name='static_image' value='https://public.tableau.com/static/images/ED/EDA-Demo/NoOutliersDashboard/1.png' />
    <param name='animate_transition' value='yes' />
    <param name='display_static_image' value='yes' />
    <param name='display_spinner' value='yes' />
    <param name='display_overlay' value='yes' />
    <param name='display_count' value='yes' />
    <param name='language' value='en-US' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1733872310940');
      var vizElement = divElement.getElementsByTagName('object')[0];
    if ( divElement.offsetWidth > 800 ) {
      vizElement.style.minWidth='420px';
      vizElement.style.maxWidth='1350px';
      vizElement.style.width='100%';
      vizElement.style.minHeight='587px';
      vizElement.style.maxHeight='887px';
      vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
  } else if ( divElement.offsetWidth > 500 ) {
      vizElement.style.minWidth='420px';
      vizElement.style.maxWidth='1350px';
      vizElement.style.width='100%';
      vizElement.style.minHeight='587px';
      vizElement.style.maxHeight='887px';
      vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
  } else {
      vizElement.style.width='100%';
      vizElement.style.height='1677px';
  }
  var scriptElement = document.createElement('script');
  scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
  vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
"""

    # Embed the Tableau visualization in Streamlit
    components.html(tableau_html, height=800)
    
   
    


# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")
    
    # Input fields for user to enter data
    
    Ward1 = st.selectbox(
        "Please select the ward information for donation bag prediction", 
        ('Clareview Ward', 'Lee Ridge Ward', 'Forest Heights Ward',
         'Silver Berry Ward', 'Crawford Plains Ward', 'Londonderry Ward',
         'Woodbend Ward', 'Blackmud Creek Ward', 'Connors Hill Ward',
         'Griesbach Ward', 'Rutherford Ward', 'Rabbit Hill Ward',
         'Namao Ward', 'Ellerslie Ward', 'Greenfield Ward',
         'Southgate Ward', 'Terwillegar Park Ward', 'Wild Rose Ward',
         'Rio Vista Ward', 'Beaumont Ward', 'Wainwright Branch'),
        key="ward_donation"
    )
    # Load encoder
    encoder = joblib.load('one_hot_encoder.pkl') 
    

    # Step 3: Transform the selected ward using the encoder
    
    ward_encoded = encoder.transform(np.array([Ward1]).reshape(-1, 1))  # Reshaping for the encoder
       
    st.write("You selected:", Ward1)
    
    routes_completed = st.slider("Routes Completed", 1, 10, 5, key="routes_completed")
    time_spent = st.slider("Time Spent (minutes)", 10, 150, 60, key="time_spent")
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 10, 2, key="adult_volunteers_bags")
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10, key="youth_volunteers_bags")
    doors_in_route = st.slider("Number of Doors in Route", 10, 2000, 100, key="doors_in_route_bags")
    
    # Predict button
    if st.button("Predict Donation Bags", key="predict_bags"):
        # Load the trained model
        model = joblib.load('best_model_bag.pkl')
     
        # Prepare input data for prediction
        input_data = np.hstack((ward_encoded, np.array([[routes_completed, time_spent, adult_volunteers, youth_volunteers, doors_in_route]])))
        
        # Display input data for verification
        st.write("You have inputted the following data for prediction:")
        st.write(f"- Ward: {Ward1}")
        st.write(f"- Routes Completed: {routes_completed}")
        st.write(f"- Time Spent: {time_spent} minutes")
        st.write(f"- Number of Adult Volunteers: {adult_volunteers}")
        st.write(f"- Number of Youth Volunteers: {youth_volunteers}")
        st.write(f"- Number of Doors: {doors_in_route}")

        # Make prediction
        prediction = model.predict(input_data)
        
        # Display the prediction
        st.success(f"The predicted number of donation bags is: {prediction[0]}")

    st.write("Enter the details to predict route completion time:") 

    # Input fields for user to enter data
    Ward2 = st.selectbox(
        "Please select the ward information for route completion prediction", 
        ('Clareview Ward', 'Lee Ridge Ward', 'Forest Heights Ward',
         'Silver Berry Ward', 'Crawford Plains Ward', 'Londonderry Ward',
         'Woodbend Ward', 'Blackmud Creek Ward', 'Connors Hill Ward',
         'Griesbach Ward', 'Rutherford Ward', 'Rabbit Hill Ward',
         'Namao Ward', 'Ellerslie Ward', 'Greenfield Ward',
         'Southgate Ward', 'Terwillegar Park Ward', 'Wild Rose Ward',
         'Rio Vista Ward', 'Beaumont Ward', 'Wainwright Branch'),
        key="ward_route"
    )
    Stake = st.selectbox(
        "Please select the stake information", 
        ('Bonnie Doon Stake', 'Gateway Stake', 'Riverbend Stake', 'Edmonton North Stake'), 
        key="stake"
    )

    # Load encoder
    encoder2 = joblib.load('onehot_encoder2.pkl') 


    # Step 3: Transform the selected ward using the encoder
    ward_encoded = encoder2.transform(np.array([Ward2, Stake]).reshape(1,-1)) # Reshaping for the encoder
    st.write("You selected:", Ward2, Stake)
    
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 10, 2, key="adult_volunteers_route")
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10, key="youth_volunteers_route")
    doors_in_route = st.slider("Number of Doors in Route", 10, 2000, 50, key="doors_in_route_route")
    
    # Predict button
    if st.button("Predict Completion Time", key="predict_route"):
        # Load the scaler and the trained model
        model2 = joblib.load('best_model_route.pkl')
        scaler = joblib.load('numerical_scaler.pkl')

        # Prepare and scale numerical features
        numerical_features2 = np.array([[adult_volunteers, youth_volunteers, doors_in_route]])
        scaled_numerical_features2 = scaler.transform(numerical_features2)
        
        # Prepare input data for prediction
        input_data2 = np.hstack((ward_encoded, scaled_numerical_features2))
        
        # Display input data for verification
        st.write("You have inputted the following data for prediction:")
        st.write(f"- Ward: {Ward2}")
        st.write(f"- Stake: {Stake}")
        st.write(f"- Number of Adult Volunteers: {adult_volunteers}")
        st.write(f"- Number of Youth Volunteers: {youth_volunteers}")
        st.write(f"- Number of Doors: {doors_in_route}")

        # Make prediction
        prediction2 = model2.predict(input_data2)
        
        # Display the prediction
        # Assuming prediction2 is an array with a float value at index 0
        st.success(f"The approximate completion time is: {int(prediction2[0])} mins")






# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Visualizations", "ML Modeling"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Visualizations":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    

if __name__ == "__main__":
    main()
