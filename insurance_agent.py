import os

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import os, urllib
import plotly.express as px
import model
import pickle

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_as_string("instructions.md"))
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Client Data Analysis")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["기존 고객 분석", "신규 고객 분석"])
    if app_mode == "신규 고객 분석":
        readme_text.empty()
        run_the_app()
    elif app_mode == "기존 고객 분석":
        readme_text.empty()
        cur_client_app()
    
# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# 신규 고객 분석 페이지
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache_data
    def load_metadata(url):
        return pd.read_csv(url,encoding='euc-kr')

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache_data
    def create_summary(metadata):
        st.write(metadata.columns)
        one_hot_encoded = pd.get_dummies(metadata[["CUST_ID", "OCCP_NAME_G"]], columns=["OCCP_NAME_G"])
        one_hot_encoded.rename(columns = lambda x: x.replace("OCCP_NAME_G_",""), inplace = True)
        return one_hot_encoded.iloc[:,2:]
    
    # 보험 해지 여부 결정
    x_resampled=pd.read_csv(r'./data/보험 해지 여부 x.csv',encoding='euc-kr')
    st.markdown(model.insurance_cancellation_predict(x_resampled.iloc[[3],1:]))

    # 대출 연체 여부 결정
    st.markdown(model.insurance_cancellation_predict(x_resampled.iloc[[3],1:]))


def cur_client_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache_data
    def load_metadata(url):
        return pd.read_csv(url,encoding='utf-8')

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache_data
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["CUST_ID", "OCCP_NAME_G"]], columns=["OCCP_NAME_G"])
        one_hot_encoded.rename(columns = lambda x: x.replace("OCCP_NAME_G_",""), inplace = True)
        return one_hot_encoded[['CUST_ID']]
    
    metadata =  load_metadata(r'./data/insurance_db.csv')
    summary = create_summary(metadata)

    select_options=frame_selector_cur_client_ui(summary)

    select_metadata = metadata[metadata['CUST_ID']==select_options][['CRDT_CARD_CNT', 'SPTCT_OCCR_MDIF', 'BNK_LNIF_AMT', 'TOT_LNIF_CNT',
       'TOT_LNIF_AMT', 'CPT_LNIF_CNT', 'SPART_LNIF_CNT', 'CPT_LNIF_AMT',
       'CRLN_OVDU_RATE', 'CRDT_OCCR_MDIF', 'CTCD_OCCR_MDIF', 'CB_GUIF_AMT',
       'TOT_CLIF_AMT', 'AUTR_FAIL_MCNT', 'TARGET']]
    
    insurance_col_df = load_metadata(r'./data/insurace_db_column.csv')
    col_index = insurance_col_df.set_index('column_name').to_dict()['column_mean']
    
    important_vars = ['CRDT_CARD_CNT', 'SPTCT_OCCR_MDIF', 'BNK_LNIF_AMT', 'TOT_LNIF_CNT',
       'TOT_LNIF_AMT', 'CPT_LNIF_CNT', 'SPART_LNIF_CNT', 'CPT_LNIF_AMT',
       'CRLN_OVDU_RATE', 'CRDT_OCCR_MDIF', 'CTCD_OCCR_MDIF', 'CB_GUIF_AMT',
       'TOT_CLIF_AMT', 'AUTR_FAIL_MCNT', 'TARGET']

    rename_dict = {}
    for key, item in col_index.items():
        if key in important_vars:
            rename_dict[key] = item

    

    select_metadata=metadata[metadata['CUST_ID']==select_options][['CRDT_CARD_CNT', 'SPTCT_OCCR_MDIF', 'BNK_LNIF_AMT', 'TOT_LNIF_CNT',
       'TOT_LNIF_AMT', 'CPT_LNIF_CNT', 'SPART_LNIF_CNT', 'CPT_LNIF_AMT',
       'CRLN_OVDU_RATE', 'CRDT_OCCR_MDIF', 'CTCD_OCCR_MDIF', 'CB_GUIF_AMT',
       'TOT_CLIF_AMT', 'AUTR_FAIL_MCNT', 'TARGET']]

    select_metadata.rename(columns=rename_dict,inplace=True)
    # 칼럼 이름 순 정렬
    select_metadata = select_metadata.reindex(sorted(select_metadata.columns), axis=1)

    # head
    u_u_col1, u_u_col2, u_u_col3, u_u_col4 = st.columns(4)
    with u_u_col1:
            st.markdown("나의 고객")
            st.markdown(f"{metadata.shape[0]}명")
    with u_u_col2:
            st.markdown("나의 기존 고객")
            st.markdown(f"{metadata.shape[0]//10*9}명")
    with u_u_col3:
            st.markdown("나의 신규 고객")
            st.markdown(f"{metadata.shape[0]//10}명")
    with u_u_col4:
            st.markdown("나의 VIP 고객")
            st.markdown(f"{metadata.shape[0]//100}명")

    
    st.markdown("----------------------------------------")


    # body
    u_col1, u_col2 = st.columns([4,1])
    with u_col1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("연체 예상 여부")
            # 본인의 연체율과 소득 대비 평균 연체율
            df_person=metadata[metadata['CUST_ID']==select_options][['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'해당 고객 보험료연체율'})
            df_group=metadata[['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'전체 보험료연체율'})
            chart_data=df_person.join(df_group).rename(index={'mean':'연체율'})
            st.bar_chart(chart_data.T,width=600,height=400,use_container_width=True)
        with col2:
            st.markdown("만기 가능 여부")
            # 본인의 완납경험횟수/(완납경험횟수+실효해지건수)와 해당 소득층 별 완납경험횟수/(완납경험횟수+실효해지건수)
            a=metadata[metadata['CUST_ID']==select_options][['FMLY_PLPY_CNT']].iloc[0][0]
            b=metadata[metadata['CUST_ID']==select_options][['CNTT_LAMT_CNT']].iloc[0][0]
            c=a/(a+b)
            a1=metadata[['FMLY_PLPY_CNT']].sum()[0]
            b1=metadata[['CNTT_LAMT_CNT']].sum()[0]
            c1=a1/(a1+b1)

            chart_data=pd.DataFrame({
                '해당 고객 만기율':[c],
                '전체고객 만기율':[c1],
            })
            st.bar_chart(chart_data.T,width=600,height=400,use_container_width=True,x=['해당 고객 만기율','전체고객 만기율'])
        with col3:
            st.markdown("현재 신용 등급")
            # 본인의 완납경험횟수/(완납경험횟수+실효해지건수)와 해당 소득층 별 완납경험횟수/(완납경험횟수+실효해지건수)
            df_person=metadata[metadata['CUST_ID']==select_options][['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'해당 고객 보험료연체율'})
            df_group=metadata[['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'전체 보험료연체율'})
            chart_data=df_person.join(df_group).rename(index={'mean':'연체율'})
            st.bar_chart(chart_data,width=600,height=400,use_container_width=True)
    with u_col2:
        st.markdown("보험료, 신용대출,  약관대출 비율")
        df_person=metadata[metadata['CUST_ID']==select_options][['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'해당 고객 보험료연체율'})
        df_group=metadata[['PREM_OVDU_RATE']].agg(['mean']).rename(columns={'PREM_OVDU_RATE':'전체 보험료연체율'})
        chart_data=df_person.join(df_group).rename(index={'mean':'연체율'})
        st.bar_chart(chart_data,width=600,height=400,use_container_width=True)

    # Uncomment these lines to peek at these DataFrames.
    # st.markdown('## 선택한 기존 고객 정보')
    # st.table(select_metadata.T)
    



# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("## Client 정보 입력")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("현재 직업을 선택하세요", summary.columns, 2)

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("실가족원수가 몇 명입니까?", 0, 100)

    # 소득 입력란
    cur_income = st.sidebar.number_input("현재 소득을 입력해주세요.",min_value=0,format='%d') / 1000

    selected_frame = ''
    return selected_frame_index, selected_frame

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_cur_client_ui(summary):
    st.sidebar.markdown("## Client 정보 입력")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("기존 고객을 선택하세요", summary['CUST_ID'].unique(), 2)
    return object_type




# Select frames based on the selection in the sidebar
@st.cache_data()
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache_resource
def get_file_content_as_string(path):
    url = 'https://github.com/HyeongseonKim/streamlit_insurance_app/blob/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def get_file_as_string(path):
    f = open(path, "r", encoding='utf-8')
    str1 =  f.read()
    print(str1)
    return str1

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache_data
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Run the YOLO model to detect objects.
def yolo_v3(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

if __name__ == "__main__":
    main()
