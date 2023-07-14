import streamlit as st
import pandas as pd

# Import for API calls
import requests

# Import for navbar
from streamlit_option_menu import option_menu

# Import for dynamic tagging
from streamlit_tags import st_tags, st_tags_sidebar

# Imports for aggrid
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

# Import for loading interactive keyboard shortcuts into the app
# from dashboard_utils.gui import keyboard_to_url
# from dashboard_utils.gui import load_keyboard_class

# 设置页面标题
st.set_page_config(page_title="Zero-Shot 文本分类", page_icon="😎")
# 设置网页图标
# st.image("logo.png", width=350)
# 设置标题
st.title("😎 Zero-Shot 文本分类")

# 设置侧栏
with st.sidebar:
    selected = option_menu(
        "",
        ["普通模式", "自费模式"],
        icons=["emoji-sunglasses", "bi-key-fill"],
        menu_icon="",
        default_index=0,
    )
    
# 判断是否有可以提交的标记    
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

# 第一个功能页面的作用
if selected == "普通模式":

    API_KEY = st.secrets["API_TOKEN"]

    API_URL = (
        "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
    )

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with st.form(key="my_form"):
        # 创建一个能够自己添加标签的输入框
        multiselectComponent = st_tags(
            label="",
            text="最多添加三个标签",
            value=["Positive", "Negative"],
            suggestions=[
                "Informational",
                "Transactional",
                "Navigational",
                "Positive",
                "Negative",
                "Neutral",
            ],
            maxtags=3,
        )

        new_line = "\n"
        nums = [
            "What a wonderful day!",
            "I lost my packet...So BAD...",
        ]

        sample = f"{new_line.join(map(str, nums))}"

        linesDeduped2 = []

        MAX_LINES = 5
        text = st.text_area(
            "输入多个短语进行分类",
            sample,
            height=400,
            key="2",
            help="最少输入 2 行短语，最多 "
            + str(MAX_LINES)
            + " 行作为一次分类输入",
        )
        lines = text.split("\n")  # A list of lines
        linesList = []
        for x in lines:
            linesList.append(x)
        linesList = list(dict.fromkeys(linesList))  # Remove dupes
        linesList = list(filter(None, linesList))  # Remove empty

        if len(linesList) > MAX_LINES:

            st.info(
                f"❄️  只有前 "
                + str(MAX_LINES)
                + " 行进行分类的过程，如果想要输入更多的数据进行分类，请使用自费模式！"
            )

        linesList = linesList[:MAX_LINES]

        submit_button = st.form_submit_button(label="提交")

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ 没有输入任何的短语！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and not multiselectComponent:
        st.warning("❄️ 你还没有添加任何分类标签！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and len(multiselectComponent) == 1:
        st.warning("❄️ 请添加至少两种不同的标签！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:
            st.session_state.valid_inputs_received = True

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            # Unhash to check status codes from the API response
            # st.write(response.status_code)
            return response.json()

        listtest = ["I want a refund", "I have a question"]
        listToAppend = []

        for row in linesList:
            output2 = query(
                {
                    "inputs": row,
                    "parameters": {"candidate_labels": multiselectComponent},
                    "options": {"wait_for_model": True},
                }
            )

            listToAppend.append(output2)

            df = pd.DataFrame.from_dict(output2)

        st.success("✅ 分类完成！")

        df = pd.DataFrame.from_dict(listToAppend)

        st.caption("")
        st.markdown("### 查看分类结果")
        st.caption("")

        # This is a list comprehension to convert the decimals to percentages
        f = [[f"{x:.2%}" for x in row] for row in df["scores"]]

        # This code is for re-integrating the labels back into the dataframe
        df["分类指数"] = f
        df.drop("scores", inplace=True, axis=1)

        # This code is to rename the columns
        df.rename(columns={"sequence": "文本短语"}, inplace=True)
        df.rename(columns={"labels": "分类标签"}, inplace=True)

        # The code below is for Ag-grid

        gb = GridOptionsBuilder.from_dataframe(df)
        # enables pivoting on all columns
        gb.configure_default_column(
            enablePivot=True, enableValue=True, enableRowGroup=True
        )
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=300,
            fit_columns_on_grid_load=False,
            configure_side_bar=True,
        )

        # The code below is for the download button

        cs, c1 = st.columns([2, 2])

        with cs:

            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode("gbk")

            csv = convert_df(df)  #

            st.download_button(
                label="导出为CSV文件",
                data=csv,
                file_name="results.csv",
                mime="text/csv",
            )
elif selected == "自费模式":

    API_KEY = st.text_input(
        "输入你的 HuggingFace API key 🤗",
        help="当你创建账号后, 你可以在以下网页找到你的token https://huggingface.co/settings/tokens",
    )
    API_URL = (
        "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
    )

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with st.form(key="my_form"):
        # 创建一个能够自己添加标签的输入框
        multiselectComponent = st_tags(
            label="",
            text="最多添加三个标签",
            value=["Positive", "Negative"],
            suggestions=[
                "Informational",
                "Transactional",
                "Navigational",
                "Positive",
                "Negative",
                "Neutral",
            ],
            maxtags=3,
        )

        new_line = "\n"
        nums = [
            "What a wonderful day!",
            "I lost my packet...So BAD...",
        ]

        sample = f"{new_line.join(map(str, nums))}"

        linesDeduped2 = []

        MAX_LINES = 5
        text = st.text_area(
            "输入多个短语进行分类",
            sample,
            height=400,
            key="2",
            help="最少输入 2 行短语",
        )
        lines = text.split("\n")  # A list of lines
        linesList = []
        for x in lines:
            linesList.append(x)
        linesList = list(dict.fromkeys(linesList))  # Remove dupes
        linesList = list(filter(None, linesList))  # Remove empty


        submit_button = st.form_submit_button(label="提交")
    if not API_KEY:
        st.warning("❄️ 没有输入你的API！")
        st.session_state.valid_inputs_received = False
        st.stop()
    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ 没有输入任何的短语！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and not multiselectComponent:
        st.warning("❄️ 你还没有添加任何分类标签！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and len(multiselectComponent) == 1:
        st.warning("❄️ 请添加至少两种不同的标签！")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:
            st.session_state.valid_inputs_received = True

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            # Unhash to check status codes from the API response
            # st.write(response.status_code)
            return response.json()

        listtest = ["I want a refund", "I have a question"]
        listToAppend = []

        for row in linesList:
            output2 = query(
                {
                    "inputs": row,
                    "parameters": {"candidate_labels": multiselectComponent},
                    "options": {"wait_for_model": True},
                }
            )

            listToAppend.append(output2)

            df = pd.DataFrame.from_dict(output2)

        st.success("✅ 分类完成！")

        df = pd.DataFrame.from_dict(listToAppend)

        st.caption("")
        st.markdown("### 查看分类结果")
        st.caption("")

        # This is a list comprehension to convert the decimals to percentages
        f = [[f"{x:.2%}" for x in row] for row in df["scores"]]

        # This code is for re-integrating the labels back into the dataframe
        df["分类指数"] = f
        df.drop("scores", inplace=True, axis=1)

        # This code is to rename the columns
        df.rename(columns={"sequence": "文本短语"}, inplace=True)
        df.rename(columns={"labels": "分类标签"}, inplace=True)

        # The code below is for Ag-grid

        gb = GridOptionsBuilder.from_dataframe(df)
        # enables pivoting on all columns
        gb.configure_default_column(
            enablePivot=True, enableValue=True, enableRowGroup=True
        )
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=300,
            fit_columns_on_grid_load=False,
            configure_side_bar=True,
        )

        # The code below is for the download button

        cs, c1 = st.columns([2, 2])

        with cs:

            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode("gbk")

            csv = convert_df(df)  #

            st.download_button(
                label="导出为CSV文件",
                data=csv,
                file_name="results.csv",
                mime="text/csv",
            )