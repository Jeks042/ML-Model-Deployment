{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8e903aaf",
   "metadata": {},
   "source": [
    "# MODEL DEPLOYMENT USING FLASK - CHUKWUJEKWU JOSEPH EZEMA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6800b66a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.0 Import Libraries\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "import pickle\n",
    "from flask import Flask, render_template, request\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "afed94b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sepal_Length</th>\n",
       "      <th>Sepal_Width</th>\n",
       "      <th>Petal_Length</th>\n",
       "      <th>Petal_Width</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>Virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>Virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Sepal_Length  Sepal_Width  Petal_Length  Petal_Width      Class\n",
       "0             5.1          3.5           1.4          0.2     Setosa\n",
       "1             4.9          3.0           1.4          0.2     Setosa\n",
       "2             4.7          3.2           1.3          0.2     Setosa\n",
       "3             4.6          3.1           1.5          0.2     Setosa\n",
       "4             5.0          3.6           1.4          0.2     Setosa\n",
       "..            ...          ...           ...          ...        ...\n",
       "145           6.7          3.0           5.2          2.3  Virginica\n",
       "146           6.3          2.5           5.0          1.9  Virginica\n",
       "147           6.5          3.0           5.2          2.0  Virginica\n",
       "148           6.2          3.4           5.4          2.3  Virginica\n",
       "149           5.9          3.0           5.1          1.8  Virginica\n",
       "\n",
       "[150 rows x 5 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 2.0 Load the Iris Data\n",
    "\n",
    "data = pd.read_csv('iris.csv')\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a32dcab4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.0 Split Data\n",
    "\n",
    "X = data.drop('Class', axis=1)\n",
    "y = data['Class']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4d2705ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 4.0 Train Model\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1f8a19d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "# 5.0 Evaluate Model\n",
    "\n",
    "y_pred = model.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print('Model Accuracy:', accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "505451f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save Model\n",
    "\n",
    "with open('model.pkl', 'wb') as file:\n",
    "    pickle.dump(model, file)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fcb913b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
