import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count, isnull, trim, lower, regexp_replace
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
import seaborn as sns


class Scalable_Health_Predictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scalable_Health_Predictor")
        self.setGeometry(100, 100, 600, 400)
        
        self.spark = SparkSession.builder.appName("Scalable Health Predictor").getOrCreate()
        self.df = None
        self.file_path = ""
        #main layout and title
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        title = QLabel("Scalable Health Predictor")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Times New Roman", 17, QFont.Bold))
        title.setStyleSheet("color: white;padding:11px;")
        main_layout.addWidget(title)

        #file upload
        fileupload = QHBoxLayout()
        self.label = QLabel("No file selected")
        self.label.setStyleSheet("padding:6px;")
        fileupload.addWidget(self.label)
        #selecting 
        self.select = QPushButton("Select CSV File")
        self.select.setStyleSheet("background-color:#4285f4;padding:9px;")
        self.select.clicked.connect(self.file_load)
        fileupload.addWidget(self.select)
        main_layout.addLayout(fileupload)
        #separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        #data cleaning
        clean = QHBoxLayout()
        self.clean_btn = QPushButton("Data Clean Process")
        self.clean_btn.setStyleSheet("background-color: #34a853; color: black;padding: 12px;")
        self.clean_btn.clicked.connect(self.clean_data)
        clean.addWidget(self.clean_btn)
        main_layout.addLayout(clean)
        #algos 
        model = QVBoxLayout()
        #logistic regression
        self.logistic_reg = QPushButton("Logistic Regression")
        self.logistic_reg.setStyleSheet("background-color: #fbbc05; color: black; padding:12px;")
        self.logistic_reg.clicked.connect(self.logistic_rega_func)
        model.addWidget(self.logistic_reg)
        #k-means clustering
        self.kmeans_cluster = QPushButton("K-Means Clustering")
        self.kmeans_cluster.setStyleSheet("background-color:violet; color: black; padding:12px;")
        self.kmeans_cluster.clicked.connect(self.kmeans_cluster_func)
        model.addWidget(self.kmeans_cluster)
        #linear reg
        self.linear_reg = QPushButton("Linear Regression")
        self.linear_reg.setStyleSheet("background-color: purple; color: black;padding:12px;")
        self.linear_reg.clicked.connect(self.linear_reg_func)
        model.addWidget(self.linear_reg)
        #random forest
        self.randomforest = QPushButton("Random Forest")
        self.randomforest.setStyleSheet("background-color: blue; color: black; padding:12px;")
        self.randomforest.clicked.connect(self.random_forest_func)
        model.addWidget(self.randomforest)
        #svm
        self.SVM = QPushButton("SVM")
        self.SVM.setStyleSheet("background-color: green; color: white;padding:12px;")
        self.SVM.clicked.connect(self.SVM_func)
        model.addWidget(self.SVM)

        main_layout.addLayout(model)
        self.setLayout(main_layout)

    def file_load(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open CSV File", "", "CSV files (*.csv)")
        if file_path:
            self.file_path = file_path
            self.label.setText(f"Loaded file: {os.path.basename(file_path)}")
        else:
            self.label.setText("No file selected")

    def clean_data(self):
        #spark reads csv that is given
        df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)
        #handling column selection, cleaning, and handling missing data/nulls

        #columns selection
        #1
        columns_selected = ['YearStart', 'LocationDesc', 'Topic', 'Question', 'Data_Value', 'Education', 'Income', 'Race/Ethnicity']
        df = df.select(columns_selected)
        #flitering of data to find relevant information about obesity
        keywords = ["Obesity", "Overweight", "BMI", "Weight Status"]
        condition = None
        for keyword in keywords:
            if condition is None:
                condition = col("Topic").contains(keyword) | col("Question").contains(keyword)
            else:
                condition |= col("Topic").contains(keyword) | col("Question").contains(keyword)
        df = df.filter(condition)
        #renaming columns for relevance and handling missing value/nulls
        df = df.withColumnRenamed("YearStart", "year").withColumnRenamed("LocationDesc", "state").withColumnRenamed("Data_Value", "obesity_rate")
        #2
        #handling missing values
        #nulls dropped
        df = df.na.drop(subset=["obesity_rate", "state"])
        df = df.fillna({"Education": "Not Reported", "Income": "Unknown", "Race/Ethnicity": "Not Specified"})
        #ensuring consistency of data
        #obesity rates not in the negatives or higher than 100
        df = df.filter((col("obesity_rate") >= 0) & (col("obesity_rate") <= 100))
        #relevalent year ranges
        df = df.filter((col("year") > 2009) & (col("year") < 2024))
        #removing duplicates just incase
        df = df.dropDuplicates()
        #changing columns/test to loweer case for better processing
        for column in ["state", "Education", "Income", "Race/Ethnicity"]:
            df = df.withColumn(column, trim(lower(regexp_replace(col(column), "[^a-zA-Z0-9 ]", ""))))

        self.df = df
        QMessageBox.information(self, "Success", "Data cleaned and preprocessed successfully.")

    def logistic_rega_func(self):
        #binary label: 1 if obesity_rate > 35 else 0
        df = self.df.withColumn("label", when(col("obesity_rate") > 35, 1).otherwise(0))
        #class weights to handle imbalance
        total = df.count()
        class_0 = df.filter(col("label") == 0).count()
        class_1 = df.filter(col("label") == 1).count()
        weight_0 = total / (2.0 * class_0)
        weight_1 = total / (2.0 * class_1)

        df = df.withColumn("weight", when(col("label") == 0, weight_0).otherwise(weight_1))

        #index categorical features
        edu_indexer = StringIndexer(inputCol="Education", outputCol="edu_index", handleInvalid="keep")
        state_indexer = StringIndexer(inputCol="state", outputCol="state_index", handleInvalid="keep")
        income_indexer = StringIndexer(inputCol="Income", outputCol="income_index", handleInvalid="keep")
        race_indexer = StringIndexer(inputCol="Race/Ethnicity", outputCol="race_index", handleInvalid="keep")

        assembler = VectorAssembler(inputCols=["edu_index", "state_index", "income_index", "race_index"], outputCol="raw_features")
        #scale features
        scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
        #logistic Regression with class weights
        lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight", maxIter=100, threshold=0.4)
        #pipeline
        pipeline = Pipeline(stages=[edu_indexer, state_indexer, income_indexer, race_indexer, assembler, scaler, lr])
        #split data
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        #train model
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        acc = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        prec = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        rec = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        metrics_msg = f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1 Score: {f1:.3f}"
        QMessageBox.information(self, "Model Performance", metrics_msg)

        #confusion Matrix
        pandas_df = predictions.select("label", "prediction").toPandas()
        cm = pd.crosstab(pandas_df["label"], pandas_df["prediction"])

        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Logistic Regression - Obesity Prediction")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        thresh = cm.values.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm.iloc[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm.iloc[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

    def kmeans_cluster_func(self):
        #load cleaned data
        df = self.df
        #index categorical features
        edu_indexer = StringIndexer(inputCol="Education", outputCol="edu_index", handleInvalid="keep")
        state_indexer = StringIndexer(inputCol="state", outputCol="state_index", handleInvalid="keep")

        assembler = VectorAssembler(inputCols=["obesity_rate", "edu_index", "state_index"], outputCol="raw_features")
        #scale features
        scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
        #KMeans
        kmeans = KMeans(k=3, seed=1, featuresCol="features", predictionCol="cluster")
        #pipeline
        pipeline = Pipeline(stages=[edu_indexer, state_indexer, assembler, scaler, kmeans])
        model = pipeline.fit(df)
        #predict clusters
        clustered_df = model.transform(df)
        #valuate clustering
        evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
        silhouette_score = evaluator.evaluate(clustered_df)

        QMessageBox.information(self, "K-Means Clustering", f"Silhouette Score: {silhouette_score:.3f}")
        #cluster summary
        cluster_summary = clustered_df.groupBy("cluster").agg(
            {"obesity_rate": "mean", "edu_index": "mean", "state_index": "mean"}
        ).orderBy("cluster")
        cluster_summary.show()

        # Plot
        pandas_df = clustered_df.select("obesity_rate", "edu_index", "cluster").toPandas()
        colors = ['red', 'green', 'blue']

        plt.figure(figsize=(10, 6))
        for cluster in sorted(pandas_df['cluster'].unique()):
            subset = pandas_df[pandas_df['cluster'] == cluster]
            jitter = np.random.uniform(-0.1, 0.1, size=len(subset))
            x_jittered = subset["edu_index"] + jitter
            plt.scatter(x_jittered, subset["obesity_rate"], label=f"Cluster {cluster}",
                    color=colors[cluster % len(colors)], alpha=0.6)

        plt.xlabel("Education Index")
        plt.ylabel("Obesity Rate")
        plt.title("K-Means Clustering")
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def linear_reg_func(self):
        #load cleaned data
        df = self.df

        #Linear Regression
        #feature encoding
        #using education vs obesity rate here
        indexer = StringIndexer(inputCol="Education", outputCol="Education_index1", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol="Education_index1", outputCol="Education_encode")

        #assembling pipline and performing linear regression
        assem = VectorAssembler(inputCols=["Education_encode"], outputCol="feats")
        linear_regression = LinearRegression(featuresCol="feats", labelCol="obesity_rate", predictionCol="predicts")

        pipeline = Pipeline(stages=[indexer, encoder, assem, linear_regression])

        #train/test data
        train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        predictions.select("Education", "obesity_rate", "predicts").show(5, truncate=False)

        #evaluation
        evaluation = RegressionEvaluator(labelCol="obesity_rate", predictionCol="predicts")
        #root mean square error
        root_mean_square_error = evaluation.setMetricName("rmse").evaluate(predictions)
        mean_absolute_error = evaluation.setMetricName("mae").evaluate(predictions)
        R2 = evaluation.setMetricName("r2").evaluate(predictions)

        lr_model = model.stages[-1]
        coef = lr_model.coefficients
        intercept = lr_model.intercept

        metrics_text = (
            f"Root Mean Square Error: {root_mean_square_error:.2f}\nMean Absolute Error: {mean_absolute_error:.2f}R² Score: {R2:.2f}\n\nCoefficients: {coef}\nIntercept: {intercept:.2f}")

        QMessageBox.information(self, "Linear Regression Results", metrics_text)

        #plotting
        switch_panadas = predictions.select("Education", "obesity_rate", "predicts").toPandas()
        grouping = switch_panadas.groupby("Education")[["obesity_rate", "predicts"]].mean().reset_index()
        grouping = grouping.sort_values("Education")

        plt.figure(figsize=(10, 6))
        plt.plot(grouping["Education"], grouping["obesity_rate"], marker="o", label="Actual", color="green")
        plt.plot(grouping["Education"], grouping["predicts"], marker="o", label="Predicted", color="skyblue")
        plt.xlabel("Education Level")
        plt.ylabel("Obesity Rate (%)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def random_forest_func(self):
        #data distribution
        df = self.df.withColumn("obesity_class",when(col("obesity_rate") < 30, "low").when(col("obesity_rate") < 35, "medium").otherwise("high"))

        #indexer 

        indexer_ed = StringIndexer(inputCol="Education", outputCol="Education_index1")
        indexer_in = StringIndexer(inputCol="Income", outputCol="Income_index1")
        indexer_ra = StringIndexer(inputCol="Race/Ethnicity", outputCol="Race_index1")
        label_indexer = StringIndexer(inputCol="obesity_class", outputCol="label")
        #assemble feature
        assembler = VectorAssembler(inputCols=["Education_index1", "Income_index1", "Race_index1"], outputCol="feats")

        #set up random forest
        random_forest = RandomForestClassifier(featuresCol="feats", labelCol="label", numTrees=200, maxDepth=20, seed=42)
        #workflow
        pipeline = Pipeline(stages=[indexer_ed, indexer_in, indexer_ra, label_indexer, assembler, random_forest])
        #split data
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        #fiting model
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        #evacualte
        evaluation = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

        accuracy = evaluation.evaluate(predictions, {evaluation.metricName: "accuracy"})
        precision = evaluation.evaluate(predictions, {evaluation.metricName: "weightedPrecision"})
        f1 = evaluation.evaluate(predictions, {evaluation.metricName: "f1"})
        recall = evaluation.evaluate(predictions, {evaluation.metricName: "weightedRecall"})

        metrics_text = (
            f"Accuracy: {accuracy:.2f}\nprecision: {precision:.2f}f1 Score: {f1:.2f}\n\recall: {recall}\n")

        QMessageBox.information(self, "Random Forest Results", metrics_text)

        #plot
        importances = model.stages[-1].featureImportances.toArray()
        feates = ["Education", "Income", "Race"]
        pd.DataFrame({"Feature": feates, "Importance": importances}).sort_values("Importance", ascending=False).plot.bar(x="Feature", y="Importance", figsize=(8, 4), color="skyblue", title="Feature Importances")
    
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def SVM_func(self):
        #getting label for obesity rate
        df = self.df.withColumn("label", when(col("obesity_rate") >= 30, 1).otherwise(0))
        #categorize education,income, race/ethincity, no continusous values
        cols = ["Education", "Income", "Race/Ethnicity"]

        indexers =[StringIndexer(inputCol=i, outputCol=f"{i}_index") for i in cols]
        encoders = [OneHotEncoder(inputCol=f"{i}_index", outputCol=f"{i}_vec") for i in cols]
        #vector format
        assembler = VectorAssembler(inputCols=[f"{i}_vec" for i in cols], outputCol="feats")
        #prep svm
        svm = LinearSVC(featuresCol="feats", labelCol="label", maxIter=10)
        #workflow  set up for model
        pipeline = Pipeline(stages=indexers + encoders + [assembler, svm])
        #data spilt
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        #train model
        model = pipeline.fit(train_data)
        #predictions made
        predictions = model.transform(test_data)
        #evalutation
        #calc meteics
        evaluation = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = evaluation.evaluate(predictions, {evaluation.metricName: "accuracy"})
        f1 = evaluation.evaluate(predictions, {evaluation.metricName: "f1"})
        precision = evaluation.evaluate(predictions, {evaluation.metricName: "weightedPrecision"})
        recall = evaluation.evaluate(predictions, {evaluation.metricName: "weightedRecall"})
        #metrics
        metrics_text = (
            f"Accuracy: {accuracy:.2f}\nprecision: {precision:.2f}f1 Score: {f1:.2f}\n\recall: {recall}\n")

        QMessageBox.information(self, "Random Forest Results", metrics_text)

        #Confusion matrix for plot
        results_pd = predictions.select("label", "prediction").toPandas()
        conf_matrix = pd.crosstab(results_pd['label'], results_pd['prediction'], rownames=['Actual'], colnames=['Predicted'])
        #plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("SVC - Confusion Matrix")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Scalable_Health_Predictor()
    window.show()
    sys.exit(app.exec_())
