日期：20230108
# Survey Methods of Kaggle Competition : Plant Seedlings Classification
## 前情提要
這篇文章記錄了我在大五下修的一堂課「數據科學裡的矩陣方法」，我大概花了一個月的時間製作，整理了學過的深度學習，還學到了很多新的方法，像是圖片資料預處理的技術、或新的建模技術。

此外，本篇文章的架構為教授提出的，以下說明架構目的：
1. Problem：提出問題與動機，盡量找別人做過的問題，或是已經被發現的問題。
2. Survey and Prelimilaries：調查網路上的其他人是如何解決這個問題的，且盡量調查多個方法，越多越好。
3. Development of Potential Improvement Skills：運用所學深入研究問題，並提出可改進別人方法的方案。
4. Comparison：觀察並比較自己提出的方法與前人有什麼不同—有算更快嗎？或是有改進準確率嗎？
5. Conclution and Remarks：試著解釋結果為何不同，或是重新回顧實作中的問題。

我認為透過這樣的架構，可以很好的檢視自己學過的知識，並應用在發現的問題上，也可以在調查前人作法時，學習到更新的技術與他人的思路。所以特別用這文章記錄我自己在實作時的心路歷程。

在這篇文章裡我會盡量復刻我查到的文章中的方法，並且比較我的實驗結果與作者的結論。
## Problem
  這是一個Kaggle的初階競賽。資料的結構很簡單，有兩個資料夾，一個train，一個test。train裡面有十二個資料夾，資料夾內有對應資料夾名稱的植物幼苗圖片，十二個資料夾共有4750張圖片。test裡面有793張圖片，我們的任務是用train裡的圖片與對應的植物名稱，來訓練模型預測test裡的植物幼苗圖片。
  
  圖片的資料是以矩陣n\*n\*3的方式儲存，其中的n是解析度，最後的3則為RGB三色數值。值得注意的是，這次的競賽給的圖片的解析度都不同，也就是每張照片的大小都不同。
## Survey and Prelimilaries
### Method 2 : Filter the Plants' Seedlings
方法一來自這篇文章，文章中提到因為植物幼苗都是綠色的，且照片中除了植物本體外，都是多種顏色的石頭或標籤紙，如果只需要辨別植物的話，應該只需要把圖片過濾出植物的部分就可以了，而且，過濾後的圖片背景全部變黑色有兩個好處，第一就是黑色會被儲存為0，想當然模型的計算會比較快，第二是儲存為0後，模型就不用處理那些七彩的石頭，複雜的顏色變單純了，準確率應該會變高。

有了上述的假設後，在原中，作者運用了cv函式庫來處理過濾綠色的工作，分別做了有過濾與沒過濾的實驗，而確實有過濾的模型表現得更好，但我自己實作卻沒有比 Method 1-1 有較好的表現。
### Method 1 :
方法二來自另一篇文章，是本篇的核心，文章中作者提出了五個步驟，但因多數內容太過艱深，所以我把它改成比較適合我的程度的五個步驟：
#### Step 1 : Analysis the Data
在做機器學習之前，一定要先觀察、分析手上的資料，本次的資料非常簡單，就是一堆圖片，大小不一，更值得一提的是品種的照片數量多寡不一。下圖是各品種照片數量的柱狀圖：

仔細觀察圖片，會發現用肉眼無法快速的分辨品種。為了比較，作者提出了t-SNE方法，用t-SNE來比喻是用機器的眼睛分辨品種，發現用機器的眼睛也無法快速地分辨。
#### Step 2 : CNN
既然用簡單的方法無法分辨，就必須提到我們的主角CNN，CNN可以有效地降低運算量，卻比單純的神經網路(NN)複雜，而不至於效能不好。一組「簡單CNN」包含三層結構：Convolution + ReLU + MaxPooling。
1. Convolution 的運算很簡單先跳過，值得一提的是其中的  Padding。我查到的解釋是因為邊緣與角落的pixel永遠不會處於 kernel 的中心，所以輸出的結果會損失一些值，感覺像邊緣被模糊掉了，又因為 kernel 的中心是最重要的，所以我們希望原圖的每一個 pixel 都會被中心經過。解決辦法就是在圖的外圍框一圈都為0的 pixels ，這樣既不會影響輸出，也不能解決上述問題。
2. ReLU 是一個非線性函數。在這邊加一個非線性函數是為了提升模型的「彈性」與「適應性」。
3. MaxPooling 會在一個區域內選出影響力最大的 pixel，可想而知輸出會在更進一步的縮小，在縮小之餘，因為我們選擇了影響力最大的 pixel，所以不至於丟失太多資訊。

接下來是第一次實驗，跑了一個小時後，我們發現訓練集的準確率高達九十九趴，但驗證集的準確率卻停只在七十幾，這就是所謂的過擬合。下圖是取自課本的定意：

此外，課本也提出了解決過擬合的方法：
第一個方法是在過擬合之前停止學習；第二個方法是增加參數和降低自由度使學習更困難。基於第二個方法，第三步我們介紹一個提高學習困難度的方法。
#### Step 3 : Image with Augmentation
過擬合可以視作我們製造的模型在背答案，因此只能在訓練集上表現好，在驗證集上（他沒看過的題目）就無法有好表現。所以我們把訓練集的資料翻轉、放大、縮小，如此一來就能增加訓練集學習的困難度，讓模型比較難背答案。
下圖是實驗結果：
可以看到兩個集的準確率都沒有脫節，且都可以達到九十多趴。為方便討論，我把這次方法與結果稱為Metho 1-1。
#### Step 4 : TransferLearning (imageNET)
第四步與第五步將會試圖改進前面的結果。
第四步可以想像為「站在巨人的肩膀上」。因為從頭開始訓練模型實在太麻煩了，耗時又需要自己想模型，如果我們能夠把別人訓練好的模型拿來用，應該會省不少時間。遷移學習就是基於上述理由的技術，下圖是keras官網的列表，裡面有可用的預訓練模型，可以依照自己需求來選擇遷移模型。我這次選MobileNET，因為他的參數最少，只有四百萬，我想我的電腦比較能負荷這樣的計算量。
#### Step 5 : Oversampling
在做訓練之前我們再使用一個資料預處理技術。回顧前面的步驟一，在柱狀圖中，我們發現每一品種的圖片量都不一樣，如果某一品種的圖片太少，可能會造成學習不佳，像是考試要考微積分，但你卻讀了一堆線性代數，所以我想要把圖片少的品種補到跟最多圖片的品種一樣，這就是過取樣技術。過取樣有很多方法，這裏只介紹一種：SMOTE。，SMOTE會從少數品種的圖片中，隨機選取一點，再計算同品種中離其「最近」的 k 個資料點，然後任意選取其中之一，兩點連線上隨機生成新資料。下圖方程式是新資料的生成公式。
經過過取樣之後，我們可以畫出柱狀圖，發現每一個品種都數量相同了。
## Development of Potential Improvement Skills
這部分我會盡力提出我自己想的解決方案。
### Method 3 : Method 2+1
方法一是過取樣再做遷移學習。方法二是做圖片綠色部分的萃取。所以我想結合兩方法的優點，就是先將圖片做綠色萃取再過取樣，最後遷移學習。下圖是學習一個小時的結果，發現並沒有改善結果。
### Method 4 : Greyscale
最後一個方法。我想因為圖片被綠色萃取後。圖片就只剩一種顏色：綠色，那是不是變成灰階也不會影響學習成果呢？況且灰階圖片只需要二維的矩陣就能表示，相較RGB需要三維矩陣少了很多，理論上計算量會減少很多。下圖是做完灰階再過取樣再遷移學習一小時的結果，我們發現沒有顯著的提升。
## Comparison
下方表格是五個方法的表現中，其中F1-score是我提交答案給kaggle的分數。
觀察過每個模型的混淆矩陣後，我發現有兩個品種的表現最差。把兩品種的圖片分別取出後，發現非常難分辨要說其一是另一個品種的局部也不為過，我想這就是表現的原因。
## Conclution and Remarks
最後我提出幾個問題自問自答：
### 為什麼ReLU比sigmoid好？
因為sigmoid的斜率只有在零附近很大，遠離零之後都很小，很小的斜率會使得模型的學習效率不佳（因為Backward Propagation 需要計算微分來更新參數，而微分也就是斜率）。反觀ReLU雖然在小於零時沒有斜率（其實也沒關係，因為Maxpool，值越小代表越不會成為特徵點），但在大於零的時候，斜率恆為一，不會因為值的大小而有變化，確保了模型的學習效率。
### 爲什麼什麼都沒做的Method1-1表現最好？
不知道。事實上我一開始是先把圖片標準化成224\*224\*3，但因為每個方法都跑很慢，一個小時只能有六十到八十趴的準確率。但當我改成150\*150\*3時效率竟然大耀進！因為要探討的細節實在太多，更進一步需要再多讀點書，並在下篇文章討論了。
### 先做綠萃取再做過取樣與先做過取樣再做綠萃取有差嗎？
我想是有差的，可以看到下圖，下圖是過取樣後我隨機挑的一個新樣本，我發現新樣本的雜訊變得很多，除了原本植物的輪廓外，還多了其他綠色雜訊，如果先做過取樣再做綠萃取，就會把這些雜訊也取出來，影響到模型表現。
### 該如何挑選適合的預訓練模型？
1. 必需先檢查這個預訓練模型的訓練集是否與我們的問題訓練集的相似度，如果相似度高的話，代表預訓練模型與我們的資料集更吻合，也更可能有好的表現。
2. 輪廓係數
## References
https://arxiv.org/pdf/1608.03983.pdf
http://www.datakit.cn/blog/2017/02/05/t_sne_full.html
https://serokell.io/blog/introduction-to-convolutional-neural-networks
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
https://towardsdatascience.com/upsampling-with-smote-for-classification-projects-e91d7c44e4bf
https://medium.com/%E6%95%B8%E5%AD%B8-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E8%88%87%E8%9F%92%E8%9B%87/smote-enn-%E8%A7%A3%E6%B1%BA%E6%95%B8%E6%93%9A%E4%B8%8D%E5%B9%B3%E8%A1%A1%E5%BB%BA%E6%A8%A1%E7%9A%84%E6%8E%A1%E6%A8%A3%E6%96%B9%E6%B3%95-cdb6324b711e
https://amielmeiseles.medium.com/how-to-choose-the-best-source-model-for-transfer-learning-41d5c91c1338
https://chtseng.wordpress.com/2018/01/19/kaggle-%E7%99%BC%E8%8A%BD%E6%A4%8D%E7%89%A9%E5%88%86%E9%A1%9E/
https://www.youtube.com/watch?v=ZCdbc9Ta1Ks
https://www.youtube.com/watch?v=NWONeJKn6kc&t=176s
https://ml001.netlify.app/pages/syllabus.html
https://learn.udacity.com/courses/ud187
https://medium.com/analytics-vidhya/image-augmentation-9b7be3972e27
https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86
https://cloud.tencent.com/developer/article/1950152
https://blog.csdn.net/playezio/article/details/80449568
https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86
