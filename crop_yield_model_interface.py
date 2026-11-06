import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px


#page set up
st.set_page_config(
    page_title="Crop Yield Prediction Model",
    page_icon="🌱",
    layout="wide"
)

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("cleaned_crop_yield_dataset.csv")
        return df
    except FileNotFoundError as e:
        st.warning(f"An Error occured: {e}")

@st.cache_resource
def load_model_artifact():
  try:
    model = joblib.load("crop_yield_prediction.pkl")
    scaler = joblib.load("scaler_features.pkl")
    feature = joblib.load("featured_columns.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    ordinal_encoder = joblib.load("ordinal_encoder.pkl")
    
    return model, scaler, feature, label_encoder, ordinal_encoder
  except FileNotFoundError as e:
    st.error(f"Model artifact not found: {e}")
    st.stop()
  except Exception as e:
    st.error(f"An error occured: {e}")
    st.stop()

@st.cache_resource
def load_rm_model_artifact():
  try:
    rm_model = joblib.load("rm_model_crop_yield_prediction.pkl")
    scaler = joblib.load("scaler_features.pkl")
    feature = joblib.load("featured_columns.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    ordinal_encoder = joblib.load("ordinal_encoder.pkl")
    
    return rm_model, scaler, feature, label_encoder, ordinal_encoder
  except FileNotFoundError as e:
    st.error(f"Model artifact not found: {e}")
    st.stop()
  except Exception as e:
    st.error(f"An error occured: {e}")
    st.stop()


# =========================
# PAGE FUNCTIONS
# =========================
# Reusable return button for analysis pages
def back_to_overview():
    if st.sidebar.button("← Back to Overview"):
        st.session_state.page = "Overview"
        st.rerun()

def format_number(value):
    if value >= 1000000000:
        return f"{value / 1000000000:.3f}B"
    elif value >= 1000000:
        return f"{value / 1000000:.1f}M"
    elif value >= 1000:
        return f"{value / 1000:.1f}K"
    else:
        return f"{value:,.0f}"

def nav():
    pages = [
            "Overview",
            "Crop Yield Data Visualization",
            "Model Interface - Linear Regression",
            "Model Interface - Random Forest Regression"
    ]
    
    selected = st.sidebar.selectbox(
        "All Pages",
        pages,
        index=pages.index(st.session_state.page),
        key="overview_nav"
        )
    
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

# OVERVIEW PAGE
def show_overview(df):
    nav()

    st.title("Crop Yield Analysis Visualization & Model Inteface")
    st.write("A comprehensive analysis of crop yield records accross countries.")

    st.markdown("---")
    
    #kpis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))

    with col2:
        most_common_item = df["item"].value_counts().idxmax() 
        st.metric("Most Common Item", most_common_item)

    with col3:
        country_highest_value = df.groupby("area")["crop_value"].sum().idxmax()
        st.metric("Country with Highest Yield", country_highest_value)

    with col4:
        avg_crp_value = df["crop_value"].mean()
        st.metric("Avg Crop Values", format_number(avg_crp_value))

    #tables for each df
    st.header("Dataset Overview")
    st.dataframe(df, width="content")


def show_dashboard(df):
    nav()

    st.title("Crop Yield Analysis Visualization")
    st.write("Visualization of the indepth analysis of the crop yield records.")
    st.markdown("---")

    # Customer-specific filters
    st.sidebar.header("Filters")

    selected_area = st.sidebar.multiselect(
        "Select Country",
        options=df["area"].unique(),
        default=df["area"].unique(),
    )

    selected_item = st.sidebar.multiselect(
        "Select Crop Item",
        options=df["item"].unique(),
        default=df["item"].unique(),
    )
    
    def filter_data(df, country, item):
        filtered_df = df[
            (df["item"].isin(item)) & 
            (df["item"].isin(item)) &
            (df["area"].isin(country)) & 
            (df["area"].isin(country))
        ]
        return filtered_df
    
    filtered_df = filter_data(df, selected_area, selected_item)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(filtered_df))

    with col2:
        total_item = filtered_df["item"].unique().count() if len(filtered_df) > 0 else 0
        st.metric("Total Crop Item", total_item)

    with col3:
        total_crop_value = filtered_df["crop_value"].sum() if len(filtered_df) > 0 else 0
        st.metric("Total Crop Values", format_number(total_crop_value))

    with col4:
        most_common_item = filtered_df["item"].value_counts().idxmax() if len(filtered_df) > 0 else 0
        st.metric("Most Common Item", most_common_item)

    # charts
    if len(filtered_df) == 0:
        st.warning('No Filter Selected. Please Adjust Your Selection.')
        return

    st.subheader('Distribution of Top 15 Countries by Crop Values')
    area_count = filtered_df.groupby("area")["crop_value"].sum().sort_values(ascending=False).head(15)
    fig1 = px.bar(
        x = area_count.index,
        y = area_count.values,
        orientation="v",
    )
    fig1.update_layout(
        width=1000,
        height=750,
        xaxis_title="Countries",
        yaxis_title= "Crop Values"
    )
    st.plotly_chart(fig1, width='stretch')


    st.subheader('Distribution of Crop Items')
    items = filtered_df["item"].value_counts()
    fig = px.pie(
        names=items.index,
        values=items.values,
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader('Distibution of Yearly Values of Crop Items')
    year_dist = df.groupby("year")["crop_value"].sum()

    fig2 = px.histogram(
        x = year_dist.index,
        y = year_dist.values,
        nbins=6
    )
    fig2.update_layout(
        height=650,
        xaxis_title="Year",
        yaxis_title= "Crop Item Values"
    )

    fig2.update_traces(
        marker_line_color="white",
        marker_line_width=1
    )
    st.plotly_chart(fig2, width='stretch')


    st.subheader("Distibution of Top 20 Countries by Rain Fall")
    area_rain = df.groupby("area")["avg_rainfall_per_year"].sum().sort_values(ascending=False).head(20)

    fig3 = px.bar(
        x = area_rain.index,
        y = area_rain.values,
        orientation="v"
    )
    fig3.update_layout(
        width=1000,
        height=750,
        xaxis_title="Countries",
        yaxis_title= "Total Rain Fall"
    )
    st.plotly_chart(fig3, width='stretch')


    st.subheader('Distribution of Crop Items')
    correlation = df[["year", "avg_rainfall_per_year", "pesticides_tonnes", "avg_temp", "crop_value"]].corr()
    fig4 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",   
        width=800,                           
        height=600
    )
    st.plotly_chart(fig4, width='stretch')


def show_interface(df):
    nav()
    #get filtered options
    def get_filtered_option(df, area=None, item=None):
        filtered_df = df.copy()
        
        if area:
            filtered_df = filtered_df[filtered_df["area"] == area]
        
        if item:
            filtered_df = filtered_df[filtered_df["item"] == item]
            
        options = {
            "years": sorted(filtered_df["year"].unique().tolist(), reverse=True),
            "countries": sorted(filtered_df["area"].unique().tolist()),
            "crop_items": sorted(filtered_df["item"].unique().tolist() if area else sorted(df["item"].unique().tolist())),
        }
        
        return options

    #Predict crop yield
    def predict_crop_yield(crop_data, model, scaler, label_encoder, ordinal_encoder, feature_columns):
        try:
            input_df = pd.DataFrame([crop_data])

            #encoding area/country
            column = "area"
            try:
                input_df[column + "_encoded"] = label_encoder[column].transform(input_df[column])
            except Exception as e:
                st.warning(f"Unknown country: {crop_data[column]}. Using default encoding.")
                return None

            # encoding crop item
            try:
                encoded_item = ordinal_encoder.transform(input_df[["item"]])
                input_df["item_encoded"] = encoded_item[:, 0]
                
            except Exception as e:
                st.warning(f"Unknown crop: {crop_data["item"]}")
                return None


            # preparing model features
            feature_dict = {
            "year": crop_data["year"],
            "area_encoded": input_df["area_encoded"].values[0],
            "avg_rainfall_per_year": crop_data["avg_rainfall_per_year"],
            "pesticides_tonnes": crop_data["pesticides_tonnes"],
            "avg_temp": crop_data["avg_temp"],
            "item_encoded": input_df["item_encoded"].values[0]
            }
            
            # arrange features in the correct order
            features = np.array([[feature_dict[column] for column in feature_columns]]) 
            
            feature_scale = scaler.transform(features)

            # predicted_value = model.predict(feature_scale)[0]
            predicted_value = float(np.asarray(model.predict(feature_scale)).flatten()[0])
            
            margin_percentage = 0.15
            min_predicted_value = predicted_value * (1 - margin_percentage)
            max_predicted_value = predicted_value * (1 + margin_percentage)
            
            return {
            "predicted_value": predicted_value,
            "min_predicted_value": min_predicted_value,
            "max_predicted_value": max_predicted_value
            }
        except Exception as e:
            # st.error(f"An error occured: {e}")
            st.exception(e)
            return None

    # STREAMLIT INTERFACE
    #---------------------
    st.title("Crop Yield Prediction")
    st.write("Enter the crop and environmental details to estimate crop yield.")
    
    model, scaler, feature, label_encoder, ordinal_encoder = load_model_artifact()
    
    df = load_dataset()
    
    #initialize session state
    if "selected_area" not in st.session_state:
        st.session_state.selected_area = None
    
    if "selected_item" not in st.session_state:
        st.session_state.selected_item = None
        
    if "show_result" not in st.session_state:
        st.session_state.show_result = False


    #OPTION SELECTIONS
    # Country Selection
    options = get_filtered_option(df)
    
    area = st.selectbox(
        "*Country",
        options=[""] + options["countries"],
        format_func=lambda x: "Select Country" if x == "" else x
    )

    # Reset crop and result when country changes
    if area and area != st.session_state.selected_area:
        st.session_state.selected_area = area
        st.session_state.selected_item = None
        st.session_state.show_result = False

    # Crop Selection
    if area:
        options = get_filtered_option(df, area=area)
    
    item = st.selectbox(
        "*Crop Item",
        options=[""] + options["crop_items"],
        format_func=lambda x: "Select Make" if x == "" else x,
        disabled=not area
    )
    
    if item and item != st.session_state.selected_item:
        st.session_state.selected_item = item
        st.session_state.selected_condition = None
        st.session_state.show_result = False

    #Year Selection
    if area and item:
        options = get_filtered_option(df, area=area, item=item)
        
    year = st.selectbox(
        "*Year",
        options=[""] + options["years"],
        format_func=lambda x: "Select Year" if x == "" else str(x),
        disabled=not (area and item)
    )

    #NUMERICAL INPUTS
    #----------------
    avg_rainfall = st.number_input(
        "Average Rainfall (mm)",
        min_value=0.0,
        value=1000.0,
        step=10.0,
        format="%.2f",
        disabled=not (area and item and year)
    )


    pesticides = st.number_input(
        "Pesticides (tonnes)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
        disabled=not (area and item and year)
    )


    avg_temp = st.number_input(
        "Average Temperature (°C)",
        value=25.0,
        step=0.1,
        format="%.2f",
        disabled=not (area and item and year)
    )

    #get result
    if st.button("Predict Crop Yield"):

        # Validate categorical selections first
        if not all([area, item, year]):

            st.warning("Please select a country, crop, and year.")

        else:

            # Create crop_data only after validation
            crop_data = {
                "area": area,
                "item": item,
                "year": int(year),
                "avg_rainfall_per_year": float(avg_rainfall),
                "pesticides_tonnes": float(pesticides),
                "avg_temp": float(avg_temp)
            }

            # Run prediction only when crop_data exists
            with st.spinner("Calculating predicted crop yield..."):

                result = predict_crop_yield(crop_data, model, scaler, label_encoder,ordinal_encoder, feature)

            # Store result
            if result is not None:

                st.session_state.show_result = True
                st.session_state.result = result
                st.session_state.crop_data = crop_data
    
    #display result
    if st.session_state.show_result and "result" in st.session_state:
        result = st.session_state.result
        crop_data = st.session_state.crop_data
        
        #display crop yield detail
        st.markdown("---")
        st.subheader(f"Estimated Crop Value: {crop_data['item']} {crop_data['area']} {crop_data['year']}")
        
        st.markdown("---")
        st.success(f"## {result['predicted_value']:,.2f}")
        
        #price range
        st.markdown("---")
        st.write(f"** Crop Value range: {result['min_predicted_value']:,.0f} - {result['max_predicted_value']:,.0f}")
        

def show_rm_interface(df):
    nav()
    # Get filtered options
    def get_filtered_option(df, area=None, item=None):

        filtered_df = df.copy()
        if area:
            filtered_df = filtered_df[filtered_df["area"] == area]

        if item:
            filtered_df = filtered_df[filtered_df["item"] == item]

        options = {
            "years": sorted(filtered_df["year"].unique().tolist()),
            "countries": sorted(filtered_df["area"].unique().tolist()),
            "crop_items": sorted(filtered_df["item"].unique().tolist())
        }

        return options


    # Predict crop yield
    def predict_crop_yield(crop_data, model, scaler, label_encoder, ordinal_encoder, feature_columns):
        try:
            input_df = pd.DataFrame([crop_data])

            #encoding area/country
            column = "area"
            try:
                input_df[column + "_encoded"] = label_encoder[column].transform(input_df[column])
            except Exception as e:
                st.warning(f"Unknown country: {crop_data[column]}. Using default encoding.")
                return None


            # encoding crop item
            try:
                encoded_item = ordinal_encoder.transform(input_df[["item"]])
                input_df["item_encoded"] = encoded_item[:, 0]
                
            except Exception as e:
                st.warning(f"Unknown crop: {crop_data["item"]}")
                return None


            # preparing model features
            feature_dict = {
            "year": crop_data["year"],
            "area_encoded": input_df["area_encoded"].values[0],
            "avg_rainfall_per_year": crop_data["avg_rainfall_per_year"],
            "pesticides_tonnes": crop_data["pesticides_tonnes"],
            "avg_temp": crop_data["avg_temp"],
            "item_encoded": input_df["item_encoded"].values[0]
            }
            
            # arrange features in the correct order
            features = np.array([[feature_dict[column] for column in feature_columns]]) 
            
            feature_scale = scaler.transform(features)


            # Random Forest prediction
            predicted_value = float(np.asarray(model.predict(feature_scale)).flatten()[0])
                        
            margin_percentage = 0.15
            min_predicted_value = predicted_value * (1 - margin_percentage)
            max_predicted_value = predicted_value * (1 + margin_percentage)
            
            return {
            "predicted_value": predicted_value,
            "min_predicted_value": min_predicted_value,
            "max_predicted_value": max_predicted_value
            }
        except Exception as e:
            # st.error(f"An error occured: {e}")
            st.exception(e)
            return None



    # -----------------------------------------
    # STREAMLIT INTERFACE
    # -----------------------------------------

    st.title("Random Forest Crop Yield Prediction")
    st.write("Enter the crop and environmental details to estimate crop yield.")


    # Load Random Forest artifacts
    rm_model, scaler, feature, label_encoder, ordinal_encoder = load_rm_model_artifact()

    df = load_dataset()


    # Initialize independent session state
    if "rf_selected_area" not in st.session_state:
        st.session_state.rf_selected_area = None

    if "rf_selected_item" not in st.session_state:
        st.session_state.rf_selected_item = None

    if "rf_show_result" not in st.session_state:
        st.session_state.rf_show_result = False


    # -----------------------------------------
    # COUNTRY SELECTION
    # -----------------------------------------

    options = get_filtered_option(df)

    area = st.selectbox(
        "Country",
        options=[""] + options["countries"],
        format_func=lambda x: (
            "Select Country" if x == "" else x
        ),
        key="rf_area"
    )


    # Reset crop and result when country changes
    if area != st.session_state.rf_selected_area:
        st.session_state.rf_selected_area = area
        st.session_state.rf_selected_item = None
        st.session_state.rf_show_result = False


    # -----------------------------------------
    # CROP SELECTION
    # -----------------------------------------

    if area:
        options = get_filtered_option(df, area=area)


    item = st.selectbox(
        "Crop Item",
        options=[""] + options["crop_items"],
        format_func=lambda x: ("Select Crop" if x == "" else x),
        disabled=not area,
        key="rf_item"
    )

    if item != st.session_state.rf_selected_item:
        st.session_state.rf_selected_item = item
        st.session_state.rf_show_result = False


    # -----------------------------------------
    # YEAR SELECTION
    # -----------------------------------------

    if area and item:
        options = get_filtered_option(df, area=area, item=item)

    year = st.selectbox(
        "Year",
        options=[""] + options["years"],
        format_func=lambda x: "Select Year" if x == "" else str(x),
        disabled=not (area and item)
    )


    # -----------------------------------------
    # NUMERICAL INPUTS
    # -----------------------------------------

    avg_rainfall = st.number_input(
        "Average Rainfall (mm)",
        min_value=0.0,
        value=1000.0,
        step=10.0,
        format="%.2f",
        disabled=not (area and item and year),
        key="rf_rainfall"
    )


    pesticides = st.number_input(
        "Pesticides (tonnes)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
        disabled=not (area and item and year),
        key="rf_pesticides"
    )


    avg_temp = st.number_input(
        "Average Temperature (°C)",
        value=25.0,
        step=0.1,
        format="%.2f",
        disabled=not (area and item and year),
        key="rf_temp"
    )


    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    if st.button("Predict Crop Yield", key="rf_predict"):

        if not all([area, item, year]):
            st.warning("Please select a country, crop, and year.")

        else:
            crop_data = {
                "area": area,
                "item": item,
                "year": int(year),
                "avg_rainfall_per_year": float(avg_rainfall),
                "pesticides_tonnes": float(pesticides),
                "avg_temp": float(avg_temp)
            }


            with st.spinner("Calculating Random Forest prediction..."):

                result = predict_crop_yield(crop_data, rm_model, scaler, label_encoder,ordinal_encoder,feature)


            if result is not None:
                st.session_state.rf_show_result = True
                st.session_state.rf_result = result
                st.session_state.rf_crop_data = crop_data


    # -----------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------

    if st.session_state.rf_show_result and "rf_result" in st.session_state:
        result = st.session_state.rf_result
        crop_data = st.session_state.rf_crop_data


        st.markdown("---")
        st.subheader(f"Estimated Crop Value: {crop_data['item']} {crop_data['area']} {crop_data['year']}")
        st.markdown("---")

        st.success(f"## {result['predicted_value']:,.2f}")


        st.markdown("---")
        st.write(f"** Crop Value range: {result['min_predicted_value']:,.0f} - {result['max_predicted_value']:,.0f}")




# =========================
# MAIN FUNCTION
# =========================

def main():
    df = load_dataset()

    # Initialize current page
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    #Page routing
    if st.session_state.page == "Overview":
        show_overview(df)

    elif st.session_state.page == "Crop Yield Data Visualization":
        show_dashboard(df)

    elif st.session_state.page == "Model Interface - Linear Regression":
        show_interface(df)

    elif st.session_state.page == "Model Interface - Random Forest Regression":
        show_rm_interface(df)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    main()