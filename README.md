This program takes a random forest and improves it with using with a gradient boost. They are combined together with a stack. 

I them compared the results with stacking GradientBoostingClassifier() and RandomForestClassifier() without any hyperparameters. This is to see how they do as they are which acts like a control. Together they got an accuracy of 0.903.
I then tested my knowledge on how the hyperparameters by using: 

gbch = GradientBoostingClassifier(n_estimators=1000, max_depth=3, learning_rate=0.01, random_state=67)

rfch = RandomForestClassifier(n_estimators=1000, random_state=67)

#^ the h is for hyperparameter

stacking = StackingClassifier(
    estimators=[('gb', gbch), ('rf', rfch)],

     # vv meta learner
    final_estimator = LogisticRegression(class_weight='balanced',
                                       C=60,
                                       solver="lbfgs", 
                                       max_iter=1000
                                       ),

    # uses cross-val(cv) predictions to train meta-learner
    cv = 4
  
  
)

stacking.fit(X_train, y_train)

print(f"GBRF Accuracy = {stacking.score(X_test, y_test):.3f}")

        #^ gradient boosted random forest

H = stacking.score(X_test, y_test)

Here H prints to 0.952 which while the difference = 0.04839. Im very please with these results. But if you see an opportunity to imporove the code please let me know. I am hungry to learn more about this field so any input is great.

Before running, make sure you have the data downloaded first
