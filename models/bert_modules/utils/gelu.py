{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOhYov2A41K1ChnZeYmMjBS",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hyunj941031/RecSys_for_Fashion_Campus_Dataset/blob/main/models/bert_modules/utils/gelu.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Puzfa0weMpaV"
      },
      "outputs": [],
      "source": [
        "import torch.nn as nn\n",
        "import torch\n",
        "import math\n",
        "\n",
        "class GELU(nn.Module):\n",
        "    def forward(self, x):\n",
        "        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))"
      ]
    }
  ]
}