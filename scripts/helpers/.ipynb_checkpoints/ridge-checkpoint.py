{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a61d2cfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def ridge(y, tx, lambda_):\n",
    "    \"\"\"implement ridge regression.\"\"\"\n",
    "    x_t = np.transpose(tx)\n",
    "    N = tx.shape[0]\n",
    "    K = tx.shape[1]\n",
    "    Id = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])\n",
    "    one = np.dot(x_t,tx) + Id\n",
    "    two = np.dot(x_t,y)\n",
    "    return np.linalg.solve(one,two)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
