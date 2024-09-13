import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import logging

from selenium.webdriver.support.wait import WebDriverWait

from Jargon.dejargonizer_data import DejargonizerData

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_logger")

# import random
# Generate a sample of 100 random integers from the range [0, 9312]
# random.seed(3)
# random_sample = random.sample(range(0, 9313), 100)
# print(random_sample)
random_sample = [3898, 8916, 2136, 6061, 7766, 1073, 215, 7687, 4249, 9024, 3839, 3141, 7704, 8863, 9005, 7804, 6506, 2467, 3799, 2484, 8571, 6388, 248, 1049, 2611, 701, 4935, 508, 4414, 7745, 6350, 6994, 6471, 7284, 2197, 5988, 1596, 587, 2227, 8108, 3555, 4226, 7146, 4932, 6900, 8310, 6322, 5749, 8750, 6677, 3807, 5517, 469, 4582, 2672, 5347, 8876, 1705, 3459, 4375, 4668, 2038, 1039, 7897, 7921, 1450, 5637, 1091, 6725, 2470, 329, 4815, 6998, 6802, 1948, 724, 736, 6189, 5422, 9025, 4572, 8280, 3865, 590, 5073, 118, 1261, 1771, 8774, 514, 3233, 6683, 4777, 4315, 2559, 695, 5567, 5141, 5901, 2266]

def answer_generator_from_csv(csv_path, indexes):
    df = pd.read_csv(csv_path)
    df = df.where(df["index"].isin(indexes)).dropna()
    return df


def extract_dejarognizer_data(soup) -> DejargonizerData:
    # /html/body/div[2]/section/div[2]/fieldset/div[1]
    stat_div = soup.findAll('fieldset')


def selenium_task(answer):
    web_driver = webdriver.Chrome()

    web_driver.get("https://scienceandpublic.com/")
    # if page == 0:
    #     # Wait until the SSO login button/link is clickable and click it
    #     sso_login_button = WebDriverWait(web_driver, 10).until(
    #         EC.element_to_be_clickable((By.LINK_TEXT, 'Sign in with AWS Management Console SSO'))
    #     )
    #     sso_login_button.click()
    text_field = WebDriverWait(web_driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="ContentTA"]'))
    )
    text_field.send_keys(answer)
    submit_button = WebDriverWait(web_driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="gradeTextBtn"]'))
    )
    submit_button.click()
    WebDriverWait(web_driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="resultsContent"]/span[1]'))
    )

    soup = BeautifulSoup(web_driver.page_source, 'html.parser')
    extract_dejarognizer_data(soup)


async def calculate_grade(answer):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, lambda: selenium_task(answer))


async def add_answer_grades(answers_df: pd.DataFrame):
    answers_df["dejargonizer_report"] = None
    tasks = [calculate_grade(row["answer"]) for index, row in answers_df.iterrows()]
    results = await asyncio.gather(*tasks)
    answers_df["dejargonizer_report"] = results
    return answers_df


async def main():

    answers = answer_generator_from_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv", random_sample)
    answers = await add_answer_grades(answers)
    answers.to_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/Jargon/llama3_18B_ask_science_jargon_index.csv")

if __name__ == "__main__":
    html = """<html><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>De-Jargonizer</title>
<link crossorigin="anonymous" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" rel="stylesheet"/>
<link href="Content/Site.css?v=c3e674f8-0c08-473c-9b3f-094241bd70f2" rel="stylesheet" type="text/css"/>
<link href="Content/Home.css?v=c4caa761-b4e8-43f6-8dcc-0a500d7fe9cc" rel="stylesheet" type="text/css"/>
<script async="" src="https://www.google-analytics.com/analytics.js" type="text/javascript"></script><script async="" src="https://www.googletagmanager.com/gtag/js?id=G-ERDBNXGFJ7&amp;l=dataLayer&amp;cx=c" type="text/javascript"></script><script async="" crossorigin="anonymous" src="https://connect.facebook.net/en_US/sdk.js?hash=6c88ecf09633989d4ea266f286269f44"></script><script id="facebook-jssdk" src="//connect.facebook.net/en_US/sdk.js#xfbml=1&amp;version=v2.10"></script><script src="../Scripts/modernizr-2.6.2.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
<script>
        $(document).ready(function () {
            var url = window.location.href;
            var indexOfSlash = url.indexOf("/");
            var subURLIndex = url.substr(indexOfSlash).indexOf("?");

            if (subURLIndex != -1)
            {
                var subURL = url.substr(subURLIndex + indexOfSlash + 1);
                var isFirstSection = $(".infoSection").index($("#" + subURL)) == 0;
                $(document).scrollTop($("#" + subURL).position().top - (isFirstSection ? 70 : 0));
            }

            var isiDevice = /ipad|iphone|ipod/i.test(navigator.userAgent.toLowerCase());

            if (isiDevice) {
                $(".dropDownElement").click(function () {
                    $(this).children(".dropdownItems").toggle();
                });
            }
        });

    </script>
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async="" src="https://www.googletagmanager.com/gtag/js?id=UA-103589501-1"></script>
<script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'UA-103589501-1');
    </script>
<style data-fbcssmodules="css:fb.css.base css:fb.css.dialog css:fb.css.iframewidget css:fb.css.customer_chat_plugin_iframe" type="text/css">.fb_hidden{position:absolute;top:-10000px;z-index:10001}.fb_reposition{overflow:hidden;position:relative}.fb_invisible{display:none}.fb_reset{background:none;border:0;border-spacing:0;color:#000;cursor:auto;direction:ltr;font-family:'lucida grande', tahoma, verdana, arial, sans-serif;font-size:11px;font-style:normal;font-variant:normal;font-weight:normal;letter-spacing:normal;line-height:1;margin:0;overflow:visible;padding:0;text-align:left;text-decoration:none;text-indent:0;text-shadow:none;text-transform:none;visibility:visible;white-space:normal;word-spacing:normal}.fb_reset>div{overflow:hidden}@keyframes fb_transform{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}.fb_animate{animation:fb_transform .3s forwards}
.fb_hidden{position:absolute;top:-10000px;z-index:10001}.fb_reposition{overflow:hidden;position:relative}.fb_invisible{display:none}.fb_reset{background:none;border:0;border-spacing:0;color:#000;cursor:auto;direction:ltr;font-family:'lucida grande', tahoma, verdana, arial, sans-serif;font-size:11px;font-style:normal;font-variant:normal;font-weight:normal;letter-spacing:normal;line-height:1;margin:0;overflow:visible;padding:0;text-align:left;text-decoration:none;text-indent:0;text-shadow:none;text-transform:none;visibility:visible;white-space:normal;word-spacing:normal}.fb_reset>div{overflow:hidden}@keyframes fb_transform{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}.fb_animate{animation:fb_transform .3s forwards}
.fb_dialog{background:rgba(82, 82, 82, .7);position:absolute;top:-10000px;z-index:10001}.fb_dialog_advanced{border-radius:8px;padding:10px}.fb_dialog_content{background:#fff;color:#373737}.fb_dialog_close_icon{background:url(https://connect.facebook.net/rsrc.php/v3/yq/r/IE9JII6Z1Ys.png) no-repeat scroll 0 0 transparent;cursor:pointer;display:block;height:15px;position:absolute;right:18px;top:17px;width:15px}.fb_dialog_mobile .fb_dialog_close_icon{left:5px;right:auto;top:5px}.fb_dialog_padding{background-color:transparent;position:absolute;width:1px;z-index:-1}.fb_dialog_close_icon:hover{background:url(https://connect.facebook.net/rsrc.php/v3/yq/r/IE9JII6Z1Ys.png) no-repeat scroll 0 -15px transparent}.fb_dialog_close_icon:active{background:url(https://connect.facebook.net/rsrc.php/v3/yq/r/IE9JII6Z1Ys.png) no-repeat scroll 0 -30px transparent}.fb_dialog_iframe{line-height:0}.fb_dialog_content .dialog_title{background:#6d84b4;border:1px solid #365899;color:#fff;font-size:14px;font-weight:bold;margin:0}.fb_dialog_content .dialog_title>span{background:url(https://connect.facebook.net/rsrc.php/v3/yd/r/Cou7n-nqK52.gif) no-repeat 5px 50%;float:left;padding:5px 0 7px 26px}body.fb_hidden{height:100%;left:0;margin:0;overflow:visible;position:absolute;top:-10000px;transform:none;width:100%}.fb_dialog.fb_dialog_mobile.loading{background:url(https://connect.facebook.net/rsrc.php/v3/ya/r/3rhSv5V8j3o.gif) white no-repeat 50% 50%;min-height:100%;min-width:100%;overflow:hidden;position:absolute;top:0;z-index:10001}.fb_dialog.fb_dialog_mobile.loading.centered{background:none;height:auto;min-height:initial;min-width:initial;width:auto}.fb_dialog.fb_dialog_mobile.loading.centered #fb_dialog_loader_spinner{width:100%}.fb_dialog.fb_dialog_mobile.loading.centered .fb_dialog_content{background:none}.loading.centered #fb_dialog_loader_close{clear:both;color:#fff;display:block;font-size:18px;padding-top:20px}#fb-root #fb_dialog_ipad_overlay{background:rgba(0, 0, 0, .4);bottom:0;left:0;min-height:100%;position:absolute;right:0;top:0;width:100%;z-index:10000}#fb-root #fb_dialog_ipad_overlay.hidden{display:none}.fb_dialog.fb_dialog_mobile.loading iframe{visibility:hidden}.fb_dialog_mobile .fb_dialog_iframe{position:sticky;top:0}.fb_dialog_content .dialog_header{background:linear-gradient(from(#738aba), to(#2c4987));border-bottom:1px solid;border-color:#043b87;box-shadow:white 0 1px 1px -1px inset;color:#fff;font:bold 14px Helvetica, sans-serif;text-overflow:ellipsis;text-shadow:rgba(0, 30, 84, .296875) 0 -1px 0;vertical-align:middle;white-space:nowrap}.fb_dialog_content .dialog_header table{height:43px;width:100%}.fb_dialog_content .dialog_header td.header_left{font-size:12px;padding-left:5px;vertical-align:middle;width:60px}.fb_dialog_content .dialog_header td.header_right{font-size:12px;padding-right:5px;vertical-align:middle;width:60px}.fb_dialog_content .touchable_button{background:linear-gradient(from(#4267B2), to(#2a4887));background-clip:padding-box;border:1px solid #29487d;border-radius:3px;display:inline-block;line-height:18px;margin-top:3px;max-width:85px;padding:4px 12px;position:relative}.fb_dialog_content .dialog_header .touchable_button input{background:none;border:none;color:#fff;font:bold 12px Helvetica, sans-serif;margin:2px -12px;padding:2px 6px 3px 6px;text-shadow:rgba(0, 30, 84, .296875) 0 -1px 0}.fb_dialog_content .dialog_header .header_center{color:#fff;font-size:16px;font-weight:bold;line-height:18px;text-align:center;vertical-align:middle}.fb_dialog_content .dialog_content{background:url(https://connect.facebook.net/rsrc.php/v3/y9/r/jKEcVPZFk-2.gif) no-repeat 50% 50%;border:1px solid #4a4a4a;border-bottom:0;border-top:0;height:150px}.fb_dialog_content .dialog_footer{background:#f5f6f7;border:1px solid #4a4a4a;border-top-color:#ccc;height:40px}#fb_dialog_loader_close{float:left}.fb_dialog.fb_dialog_mobile .fb_dialog_close_icon{visibility:hidden}#fb_dialog_loader_spinner{animation:rotateSpinner 1.2s linear infinite;background-color:transparent;background-image:url(https://connect.facebook.net/rsrc.php/v3/yD/r/t-wz8gw1xG1.png);background-position:50% 50%;background-repeat:no-repeat;height:24px;width:24px}@keyframes rotateSpinner{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.fb_iframe_widget{display:inline-block;position:relative}.fb_iframe_widget span{display:inline-block;position:relative;text-align:justify}.fb_iframe_widget iframe{position:absolute}.fb_iframe_widget_fluid_desktop,.fb_iframe_widget_fluid_desktop span,.fb_iframe_widget_fluid_desktop iframe{max-width:100%}.fb_iframe_widget_fluid_desktop iframe{min-width:220px;position:relative}.fb_iframe_widget_lift{z-index:1}.fb_iframe_widget_fluid{display:inline}.fb_iframe_widget_fluid span{width:100%}
.fb_mpn_mobile_landing_page_slide_out{animation-duration:200ms;animation-name:fb_mpn_landing_page_slide_out;transition-timing-function:ease-in}.fb_mpn_mobile_landing_page_slide_out_from_left{animation-duration:200ms;animation-name:fb_mpn_landing_page_slide_out_from_left;transition-timing-function:ease-in}.fb_mpn_mobile_landing_page_slide_up{animation-duration:500ms;animation-name:fb_mpn_landing_page_slide_up;transition-timing-function:ease-in}.fb_mpn_mobile_bounce_in{animation-duration:300ms;animation-name:fb_mpn_bounce_in;transition-timing-function:ease-in}.fb_mpn_mobile_bounce_out{animation-duration:300ms;animation-name:fb_mpn_bounce_out;transition-timing-function:ease-in}.fb_mpn_mobile_bounce_out_v2{animation-duration:300ms;animation-name:fb_mpn_fade_out;transition-timing-function:ease-in}.fb_customer_chat_bounce_in_v2{animation-duration:300ms;animation-name:fb_bounce_in_v2;transition-timing-function:ease-in}.fb_customer_chat_bounce_in_from_left{animation-duration:300ms;animation-name:fb_bounce_in_from_left;transition-timing-function:ease-in}.fb_customer_chat_bounce_out_v2{animation-duration:300ms;animation-name:fb_bounce_out_v2;transition-timing-function:ease-in}.fb_customer_chat_bounce_out_from_left{animation-duration:300ms;animation-name:fb_bounce_out_from_left;transition-timing-function:ease-in}.fb_invisible_flow{display:inherit;height:0;overflow-x:hidden;width:0}@keyframes fb_mpn_landing_page_slide_out{0%{margin:0 12px;width:100% - 24px}60%{border-radius:18px}100%{border-radius:50%;margin:0 24px;width:60px}}@keyframes fb_mpn_landing_page_slide_out_from_left{0%{left:12px;width:100% - 24px}60%{border-radius:18px}100%{border-radius:50%;left:12px;width:60px}}@keyframes fb_mpn_landing_page_slide_up{0%{bottom:0;opacity:0}100%{bottom:24px;opacity:1}}@keyframes fb_mpn_bounce_in{0%{opacity:.5;top:100%}100%{opacity:1;top:0}}@keyframes fb_mpn_fade_out{0%{bottom:30px;opacity:1}100%{bottom:0;opacity:0}}@keyframes fb_mpn_bounce_out{0%{opacity:1;top:0}100%{opacity:.5;top:100%}}@keyframes fb_bounce_in_v2{0%{opacity:0;transform:scale(0, 0);transform-origin:bottom right}50%{transform:scale(1.03, 1.03);transform-origin:bottom right}100%{opacity:1;transform:scale(1, 1);transform-origin:bottom right}}@keyframes fb_bounce_in_from_left{0%{opacity:0;transform:scale(0, 0);transform-origin:bottom left}50%{transform:scale(1.03, 1.03);transform-origin:bottom left}100%{opacity:1;transform:scale(1, 1);transform-origin:bottom left}}@keyframes fb_bounce_out_v2{0%{opacity:1;transform:scale(1, 1);transform-origin:bottom right}100%{opacity:0;transform:scale(0, 0);transform-origin:bottom right}}@keyframes fb_bounce_out_from_left{0%{opacity:1;transform:scale(1, 1);transform-origin:bottom left}100%{opacity:0;transform:scale(0, 0);transform-origin:bottom left}}@keyframes slideInFromBottom{0%{opacity:.1;transform:translateY(100%)}100%{opacity:1;transform:translateY(0)}}@keyframes slideInFromBottomDelay{0%{opacity:0;transform:translateY(100%)}97%{opacity:0;transform:translateY(100%)}100%{opacity:1;transform:translateY(0)}}</style></head>
<body>
<div class="navbar navbar-inverse navbar-fixed-top">
<div class="container">
<div class="navbar-header">
<button class="navbar-toggle" data-target=".navbar-collapse" data-toggle="collapse" type="button">
<span class="icon-bar"></span>
<span class="icon-bar"></span>
<span class="icon-bar"></span>
</button>
</div>
<div class="navbar-collapse collapse">
<ul class="nav navbar-nav">
<li><a class="navbar-brand" href="/">De-Jargonizer</a></li>
<li><a class="navbar-brand" href="/GroupGrading">Multiple Texts</a></li>
<li><a class="navbar-brand highlight" href="/HalfLife">Half Life</a></li>
<li><a class="navbar-brand" href="/Hebrew">Hebrew</a></li>
<li class="dropDownElement">
<a class="dropDownBtn">About</a>
<div class="dropdownItems">
<a href="../Home/About?whatIsIt">What is it?</a>
<a href="../Home/About?whyDoWeNeedIt">Why do we need it?</a>
<a href="../Home/About?whoShouldUseIt">Who should use it?</a>
<a href="../Home/About?inTheNews">In the news</a>
<a href="../Home/About?buildYourOwnDeJargonizer">Build your own De-Jargonizer</a>
</div>
</li>
<li class="dropDownElement">
<a class="dropDownBtn">Instructions</a>
<div class="dropdownItems">
<a href="../Home/Instructions?howToUseIt">How to use it?</a>
<a href="../Home/Instructions?fileTypes">File types</a>
<a href="../Home/Instructions?judgingResults">Judging results</a>
<a href="../Home/Instructions?howToReadResults">How to read results?</a>
</div>
</li>
<li class="dropDownElement">
<a class="dropDownBtn">Development</a>
<div class="dropdownItems">
<a href="../Home/Development?wordFrequencyLevels">Word frequency levels</a>
<a href="../Home/Development?development">Development</a>
<a href="../Home/Development?Developers">Developers</a>
</div>
</li>
<li class="dropDownElement">
<a class="dropDownBtn">How to cite</a>
<div class="dropdownItems">
<a href="../Home/HowToCite?howToCite">How to cite?</a>
</div>
</li>
<li class="dropDownElement">
<a class="dropDownBtn">Contact Us</a>
<div class="dropdownItems">
<a href="../Home/ContactUs?contact">Contact</a>
</div>
</li>
</ul>
</div>
</div>
</div>
<div class="container body-content">
<script>
				$(document).ready(function () {
					$(document).scrollTop($('#resultDiv').position().top);
				});
			</script>
<link href="Content/TextGrading.css?v=098c0a02-8715-4475-ade8-325aa6695219" rel="stylesheet" type="text/css"/>
<section id="textGradingSection">
<div class="fb_reset" id="fb-root"><div style="position: absolute; top: -10000px; width: 0px; height: 0px;"><div></div></div></div>
<script>
		(function (d, s, id) {
			var js, fjs = d.getElementsByTagName(s)[0];
			if (d.getElementById(id)) return;
			js = d.createElement(s); js.id = id;
			js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.10";
			fjs.parentNode.insertBefore(js, fjs);
		}(document, 'script', 'facebook-jssdk'));</script>
<h1 class="sectionTitle">De-Jargonizer</h1>
<p class="textGradingSubTitle">How accessible is your work? Paste your article or upload a file to analyze the amount of jargon in your writing.</p>
<p class="textGradingSubTitle"><img src="/Content/Assets/new-stamp.jpg" width="40"/> Check out our new <a href="/HalfLife">Half life writing exercise</a> for distilling your message.</p>
<p class="textGradingSubTitle"><img src="/Content/Assets/new-stamp.jpg" width="40"/> Join our new and <a href="http://edx.org/course/science-communication">free online science communication course</a> on edX!</p>
<form action="/" enctype="multipart/form-data" id="textGradingForm" method="post"> <fieldset>
<legend>Time Period</legend>
<select id="timePriodDDL" name="timePriodDDL">
<option value="English2012_2015">2012 - 2015</option>
<option value="English2013_2016">2013 - 2016</option>
<option value="English2014_2017">2014 - 2017</option>
<option value="English2015_2018">2015 - 2018</option>
<option value="English2016_2019">2016 - 2019</option>
<option value="English2017_2020">2017 - 2020</option>
<option selected="selected" value="English2018_2021">2018 - 2021</option>
</select>
</fieldset>
<fieldset>
<legend>Article <span style="font-size: small">(max file size is 15MB)</span></legend>
<input draggable="true" id="ArticleFU" name="ArticleFU" placeholder="Hello" type="file" value="HH"/>
<p id="textManual">You can also insert the text manually:</p>
<textarea class="input-group-sm" id="ContentTA" name="ContentTA"></textarea>
</fieldset>
<fieldset>
<button class="btn btn-primary btn-block btn-lg" id="gradeTextBtn" type="submit">Start</button>
</fieldset>
</form>
<div id="resultDiv">
<fieldset>
<legend>Result</legend>
<div id="statDiv">
<div id="resultGraph">
<img id="resultScala" src="/TextGrading/Scala?score=7"/>
</div>
<div>
<span class="resultSpan">Common:</span>
<span class="stat">81%, 203</span>
</div>
<div>
<span class="resultSpan">Mid-Frequency:</span>
<span class="stat">12%, 31</span>
</div>
<div>
<span class="resultSpan">Rare:</span>
<span class="stat">7%, 18</span>
</div>
<div>
<span class="resultSpan" id="scoreSpan">Suitability for general audience score:</span>
<span id="scoreExplanation">
						A total score was defined as 0-100. If all the words in the text are common, the text score is 100; each mid-frequency or jargon word reduces the score. The score is based on the following equation:
						<img id="equationImg" src="../Content/Assets/scoreEquation.png"/>
</span>
<span class="stat">87</span>
</div>
<div>
<span class="resultSpan">Number Of Words:</span>
<span class="stat">252</span>
</div>
<div>
<form action="/TextGrading/Download" enctype="multipart/form-data" id="downloadForm" method="post"> <input class="btn btn-primary btn-block btn-lg" id="downloadBtn" name="downloadBtn" type="submit" value="Download"/>
</form>
<div class="fb-share-button fb_iframe_widget" data-href="http://scienceandpublic.com/TextGrading/ImageResult?commonScore=81%, 203&amp;midScore=12%, 31&amp;rareScore=7%, 18&amp;totalScore=87&amp;numberOfWords=252" data-layout="button" data-mobile-iframe="true" data-size="large" fb-iframe-plugin-query="app_id=&amp;container_width=220&amp;href=http%3A%2F%2Fscienceandpublic.com%2FTextGrading%2FImageResult%3FcommonScore%3D81%25%2C%2520203%26midScore%3D12%25%2C%252031%26rareScore%3D7%25%2C%252018%26totalScore%3D87%26numberOfWords%3D252&amp;layout=button&amp;locale=en_US&amp;mobile_iframe=true&amp;sdk=joey&amp;size=large" fb-xfbml-state="parsed"><span style="vertical-align: top; width: 0px; height: 0px; overflow: hidden;"><iframe allow="encrypted-media" allowfullscreen="true" allowtransparency="true" data-testid="fb:share_button Facebook Social Plugin" frameborder="0" height="1000px" name="fd8ed21cc37578904" scrolling="no" src="https://www.facebook.com/v2.10/plugins/share_button.php?app_id=&amp;channel=https%3A%2F%2Fstaticxx.facebook.com%2Fx%2Fconnect%2Fxd_arbiter%2F%3Fversion%3D46%23cb%3Df186ebdfd69cf0f6a%26domain%3Dscienceandpublic.com%26is_canvas%3Dfalse%26origin%3Dhttps%253A%252F%252Fscienceandpublic.com%252Ff561712b34f3f8240%26relation%3Dparent.parent&amp;container_width=220&amp;href=http%3A%2F%2Fscienceandpublic.com%2FTextGrading%2FImageResult%3FcommonScore%3D81%25%2C%2520203%26midScore%3D12%25%2C%252031%26rareScore%3D7%25%2C%252018%26totalScore%3D87%26numberOfWords%3D252&amp;layout=button&amp;locale=en_US&amp;mobile_iframe=true&amp;sdk=joey&amp;size=large" style="border: none; visibility: hidden;" title="fb:share_button Facebook Social Plugin" width="1000px"></iframe></span></div>
</div>
</div>
<div class="input-group-sm" id="resultsContent" name="resultsContent"><span class="commonWord">Let's</span><span class="commonWord"> </span><span class="normalWord">dive</span><span class="commonWord"> </span><span class="commonWord">into</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">world</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">tiny</span><span class="commonWord"> </span><span class="normalWord">projectiles</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">Imagine</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="commonWord">game</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="normalWord">cosmic</span><span class="commonWord"> </span><span class="rareWord">dodgeball</span><span class="commonWord"> </span><span class="commonWord">where</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">"</span><span class="commonWord">"</span><span class="normalWord">balls</span><span class="commonWord">"</span><span class="commonWord">"</span><span class="commonWord"> </span><span class="commonWord">are</span><span class="commonWord"> </span><span class="normalWord">atoms</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">and</span><span class="commonWord"> </span><span class="commonWord">we're</span><span class="commonWord"> </span><span class="commonWord">trying</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">understand</span><span class="commonWord"> </span><span class="commonWord">what</span><span class="commonWord"> </span><span class="commonWord">happens</span><span class="commonWord"> </span><span class="commonWord">when</span><span class="commonWord"> </span><span class="commonWord">one</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">these</span><span class="commonWord"> </span><span class="rareWord">minuscule</span><span class="commonWord"> </span><span class="normalWord">bullets</span><span class="commonWord"> </span><span class="commonWord">is</span><span class="commonWord"> </span><span class="commonWord">fired</span><span class="commonWord"> </span><span class="commonWord">at</span><span class="commonWord"> </span><span class="commonWord">us</span><span class="commonWord">.</span><span class="commonWord"><br/></span><span class="commonWord"><br/></span><span class="commonWord">The</span><span class="commonWord"> </span><span class="commonWord">key</span><span class="commonWord"> </span><span class="normalWord">concept</span><span class="commonWord"> </span><span class="commonWord">here</span><span class="commonWord"> </span><span class="commonWord">is</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">size</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="normalWord">relative</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">target</span><span class="commonWord"> </span><span class="commonWord">(</span><span class="commonWord">you</span><span class="commonWord">)</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="normalWord">Atoms</span><span class="commonWord"> </span><span class="commonWord">are</span><span class="commonWord"> </span><span class="commonWord">incredibly</span><span class="commonWord"> </span><span class="commonWord">small</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">with</span><span class="commonWord"> </span><span class="normalWord">sizes</span><span class="commonWord"> </span><span class="normalWord">measured</span><span class="commonWord"> </span><span class="commonWord">in</span><span class="commonWord"> </span><span class="rareWord">picometers</span><span class="commonWord"> </span><span class="commonWord">(</span><span class="commonWord">1</span><span class="commonWord"> </span><span class="rareWord">picometer</span><span class="commonWord"> </span><span class="commonWord">=</span><span class="commonWord"> </span><span class="commonWord">0</span><span class="commonWord">.</span><span class="commonWord">0</span><span class="commonWord">0</span><span class="commonWord">0</span><span class="commonWord">0</span><span class="commonWord">0</span><span class="commonWord">1</span><span class="commonWord"> </span><span class="rareWord">millimeters</span><span class="commonWord">)</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">For</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">hit</span><span class="commonWord"> </span><span class="commonWord">you</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">it</span><span class="commonWord"> </span><span class="commonWord">needs</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">be</span><span class="commonWord"> </span><span class="commonWord">large</span><span class="commonWord"> </span><span class="commonWord">enough</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="normalWord">interact</span><span class="commonWord"> </span><span class="commonWord">significantly</span><span class="commonWord"> </span><span class="commonWord">with</span><span class="commonWord"> </span><span class="commonWord">your</span><span class="commonWord"> </span><span class="commonWord">body</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">Think</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">it</span><span class="commonWord"> </span><span class="commonWord">like</span><span class="commonWord"> </span><span class="commonWord">trying</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">hit</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="commonWord">building</span><span class="commonWord"> </span><span class="commonWord">with</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="rareWord">pebble</span><span class="commonWord"> </span><span class="commonWord">–</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="rareWord">pebble</span><span class="commonWord"> </span><span class="commonWord">will</span><span class="commonWord"> </span><span class="commonWord">likely</span><span class="commonWord"> </span><span class="normalWord">bounce</span><span class="commonWord"> </span><span class="commonWord">off</span><span class="commonWord"> </span><span class="commonWord">or</span><span class="commonWord"> </span><span class="commonWord">get</span><span class="commonWord"> </span><span class="commonWord">lost</span><span class="commonWord"> </span><span class="commonWord">in</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="normalWord">cracks</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">but</span><span class="commonWord"> </span><span class="commonWord">if</span><span class="commonWord"> </span><span class="commonWord">we</span><span class="commonWord"> </span><span class="commonWord">scale</span><span class="commonWord"> </span><span class="commonWord">up</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="normalWord">baseball</span><span class="commonWord">-</span><span class="normalWord">sized</span><span class="commonWord"> </span><span class="normalWord">object</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="normalWord">it'll</span><span class="commonWord"> </span><span class="commonWord">definitely</span><span class="commonWord"> </span><span class="commonWord">make</span><span class="commonWord"> </span><span class="commonWord">an</span><span class="commonWord"> </span><span class="commonWord">impact</span><span class="commonWord">.</span><span class="commonWord"><br/></span><span class="commonWord"><br/></span><span class="commonWord">Now</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">let's</span><span class="commonWord"> </span><span class="commonWord">consider</span><span class="commonWord"> </span><span class="commonWord">what</span><span class="commonWord"> </span><span class="commonWord">happens</span><span class="commonWord"> </span><span class="commonWord">when</span><span class="commonWord"> </span><span class="commonWord">an</span><span class="commonWord"> </span><span class="rareWord">atom</span><span class="commonWord">-</span><span class="normalWord">sized</span><span class="commonWord"> </span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="commonWord">is</span><span class="commonWord"> </span><span class="commonWord">fired</span><span class="commonWord"> </span><span class="commonWord">at</span><span class="commonWord"> </span><span class="commonWord">you</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">In</span><span class="commonWord"> </span><span class="commonWord">this</span><span class="commonWord"> </span><span class="commonWord">scenario</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="commonWord">would</span><span class="commonWord"> </span><span class="commonWord">essentially</span><span class="commonWord"> </span><span class="commonWord">be</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="commonWord">single</span><span class="commonWord"> </span><span class="rareWord">atom</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">like</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="rareWord">helium</span><span class="commonWord"> </span><span class="rareWord">nucleus</span><span class="commonWord"> </span><span class="commonWord">(</span><span class="commonWord">2</span><span class="commonWord"> </span><span class="rareWord">protons</span><span class="commonWord"> </span><span class="commonWord">and</span><span class="commonWord"> </span><span class="commonWord">2</span><span class="commonWord"> </span><span class="rareWord">neutrons</span><span class="commonWord">)</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">Due</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">its</span><span class="commonWord"> </span><span class="rareWord">minuscule</span><span class="commonWord"> </span><span class="commonWord">size</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="rareWord">atom</span><span class="commonWord">-</span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="commonWord">would</span><span class="commonWord"> </span><span class="commonWord">likely</span><span class="commonWord"> </span><span class="commonWord">pass</span><span class="commonWord"> </span><span class="commonWord">right</span><span class="commonWord"> </span><span class="commonWord">through</span><span class="commonWord"> </span><span class="commonWord">your</span><span class="commonWord"> </span><span class="commonWord">body</span><span class="commonWord"> </span><span class="commonWord">without</span><span class="commonWord"> </span><span class="commonWord">causing</span><span class="commonWord"> </span><span class="commonWord">any</span><span class="commonWord"> </span><span class="commonWord">significant</span><span class="commonWord"> </span><span class="commonWord">damage</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">This</span><span class="commonWord"> </span><span class="commonWord">is</span><span class="commonWord"> </span><span class="commonWord">because</span><span class="commonWord"> </span><span class="normalWord">atoms</span><span class="commonWord"> </span><span class="commonWord">are</span><span class="commonWord"> </span><span class="commonWord">so</span><span class="commonWord"> </span><span class="commonWord">small</span><span class="commonWord"> </span><span class="commonWord">that</span><span class="commonWord"> </span><span class="commonWord">they</span><span class="commonWord"> </span><span class="commonWord">can</span><span class="commonWord"> </span><span class="commonWord">easily</span><span class="commonWord"> </span><span class="commonWord">fit</span><span class="commonWord"> </span><span class="commonWord">between</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="normalWord">molecules</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">your</span><span class="commonWord"> </span><span class="commonWord">skin</span><span class="commonWord"> </span><span class="commonWord">or</span><span class="commonWord"> </span><span class="commonWord">even</span><span class="commonWord"> </span><span class="normalWord">penetrate</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">spaces</span><span class="commonWord"> </span><span class="commonWord">within</span><span class="commonWord"> </span><span class="commonWord">your</span><span class="commonWord"> </span><span class="commonWord">cells</span><span class="commonWord">.</span><span class="commonWord"> </span><span class="commonWord">To</span><span class="commonWord"> </span><span class="commonWord">give</span><span class="commonWord"> </span><span class="commonWord">you</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="commonWord">better</span><span class="commonWord"> </span><span class="commonWord">idea</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">imagine</span><span class="commonWord"> </span><span class="commonWord">trying</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">hit</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="normalWord">brick</span><span class="commonWord"> </span><span class="commonWord">wall</span><span class="commonWord"> </span><span class="commonWord">with</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="commonWord">single</span><span class="commonWord"> </span><span class="normalWord">grain</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="normalWord">sand</span><span class="commonWord"> </span><span class="commonWord">–</span><span class="commonWord"> </span><span class="commonWord">it's</span><span class="commonWord"> </span><span class="commonWord">not</span><span class="commonWord"> </span><span class="commonWord">going</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">make</span><span class="commonWord"> </span><span class="commonWord">much</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="commonWord">an</span><span class="commonWord"> </span><span class="commonWord">impact</span><span class="commonWord">!</span><span class="commonWord"> </span><span class="commonWord">For</span><span class="commonWord"> </span><span class="commonWord">a</span><span class="commonWord"> </span><span class="normalWord">bullet</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">actually</span><span class="commonWord"> </span><span class="commonWord">hit</span><span class="commonWord"> </span><span class="commonWord">you</span><span class="commonWord"> </span><span class="commonWord">and</span><span class="commonWord"> </span><span class="commonWord">cause</span><span class="commonWord"> </span><span class="commonWord">damage</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">it</span><span class="commonWord"> </span><span class="commonWord">would</span><span class="commonWord"> </span><span class="commonWord">need</span><span class="commonWord"> </span><span class="commonWord">to</span><span class="commonWord"> </span><span class="commonWord">be</span><span class="commonWord"> </span><span class="commonWord">significantly</span><span class="commonWord"> </span><span class="commonWord">larger</span><span class="commonWord">,</span><span class="commonWord"> </span><span class="commonWord">on</span><span class="commonWord"> </span><span class="commonWord">the</span><span class="commonWord"> </span><span class="commonWord">order</span><span class="commonWord"> </span><span class="commonWord">of</span><span class="commonWord"> </span><span class="rareWord">micrometers</span><span class="commonWord"> </span><span class="commonWord">(</span><span class="commonWord">1</span><span class="commonWord"> </span><span class="rareWord">micrometer</span><span class="commonWord"> </span><span class="commonWord">=</span><span class="commonWord"> </span><span class="commonWord">0</span><span class="commonWord">.</span><span class="commonWord">0</span><span class="commonWord">0</span><span class="commonWord">1</span><span class="commonWord"> </span><span class="rareWord">millimeters</span><span class="commonWord">)</span><span class="commonWord"> </span><span class="commonWord">or</span><span class="commonWord"> </span><span class="commonWord">even</span><span class="commonWord"> </span><span class="commonWord">larger</span><span class="commonWord">.</span></div>
</fieldset>
</div>
</section>
<hr/>
<footer>
<p>© 2024 - Jargon Project</p>
<p>visits: 49941</p>
</footer>
</div>
<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
<script crossorigin="anonymous" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</body></html>"""
    soup = BeautifulSoup(html, 'html.parser')

    extract_dejarognizer_data(html)
    # asyncio.run(main())
    # selenium_task("""Let's dive into the world of tiny projectiles. Imagine a game of cosmic dodgeball where the ""balls"" are atoms, and we're trying to understand what happens when one of these minuscule bullets is fired at us.
#
# The key concept here is the size of the bullet relative to the target (you). Atoms are incredibly small, with sizes measured in picometers (1 picometer = 0.000001 millimeters). For a bullet to hit you, it needs to be large enough to interact significantly with your body. Think of it like trying to hit a building with a pebble – the pebble will likely bounce off or get lost in the cracks, but if we scale up to a baseball-sized object, it'll definitely make an impact.
#
# Now, let's consider what happens when an atom-sized bullet is fired at you. In this scenario, the bullet would essentially be a single atom, like a helium nucleus (2 protons and 2 neutrons). Due to its minuscule size, the atom-bullet would likely pass right through your body without causing any significant damage. This is because atoms are so small that they can easily fit between the molecules of your skin or even penetrate the spaces within your cells. To give you a better idea, imagine trying to hit a brick wall with a single grain of sand – it's not going to make much of an impact! For a bullet to actually hit you and cause damage, it would need to be significantly larger, on the order of micrometers (1 micrometer = 0.001 millimeters) or even larger.""")