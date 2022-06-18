import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def assess_classifiers(X, y, preproc=False):
        """This function requires an X (features)
        and a y (target), and returns a DataFrame with
        the names of classifiers, and their baseline score
        of predicting y given x.
        """

        try:
            X.shape[1]
        except IndexError:
            X = X.reshape(-1, 1)

        if len(set(y)) == 1:
            return "Cannot use target with only one class."

        # instantiate different classifiers
        k_regressor = KNeighborsClassifier()
        s_model = SVC(random_state=42)
        classifier = DecisionTreeClassifier(random_state=42)
        random_classifier = RandomForestClassifier(random_state=42)
        ada = AdaBoostClassifier(random_state=42)
        gradient = GradientBoostingClassifier(random_state=42)

        # calculate the cross validation score for each model
        score_k = cross_val_score(k_regressor, X, y, cv=5)
        score_s = cross_val_score(s_model, X, y, cv=5)
        score_c = cross_val_score(classifier, X, y, cv=5)
        score_rc = cross_val_score(random_classifier, X, y, cv=5)
        score_a = cross_val_score(ada, X, y, cv=5)
        score_g = cross_val_score(gradient, X, y, cv=5)

        # create a DataFrame that contains the name and score of each model
        scores = pd.DataFrame(columns=['scores'], index=['KNeighbors', 'Support Vector', 'Decision Tree',
                                                'Random Forest', 'Ada Boost', 'Gradient Boost'],
                      data=[score_k.mean(), score_s.mean(), score_c.mean(), score_rc.mean(),
                           score_a.mean(), score_g.mean()])
        return scores
