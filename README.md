# Cuisine Classification using Decision Trees

![Decision Tree](https://raw.githubusercontent.com/EnricGE/Cuisine-Classification-using-Decision-Trees/main/results/Decision-Tree-for-Cuisine-Classification.png)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Science Process](#data-science-process)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

## 🍲 Overview
This project applies machine learning techniques to classify culinary recipes into their respective cuisines based solely on ingredient information. Using a dataset compiled by researcher Yong-Yeol Ahn, I built decision tree models that can identify patterns in ingredient usage across different Asian and Indian cuisines with approximately 70% accuracy.

The classification of cuisines has practical applications ranging from recipe recommendation systems to culinary innovation and cultural studies.

## ✨ Features
- Classification of recipes into Korean, Japanese, Chinese, Thai, and Indian cuisines
- Identification of distinctive ingredients for each cuisine
- Analysis of cuisine similarities based on ingredient patterns
- Visualizations of decision trees, confusion matrices, and ingredient heatmaps
- Comprehensive evaluation of model performance

## 🔬 Data Science Process

### Data Preparation
- **Data Source**: Comprehensive recipe dataset compiled from multiple recipe websites
- **Cleaning Process**:
  - Standardized cuisine names
  - Converted categorical variables to binary
  - Filtered cuisines with insufficient samples
- **Final Dataset**: Focused on Korean, Japanese, Chinese, Thai, and Indian cuisines

### Data Modeling
- Created decision tree classifier named `bamboo_tree`
- Optimized tree depth through experimentation:
  - Initial exploration: max_depth=3 (for interpretability)
  - Final model: max_depth=15 (for performance)
- Used stratified sampling for balanced testing

### Data Visualization
The project includes several visualizations:
- Decision tree diagram showing classification rules
- Confusion matrix for model performance analysis
- Ingredient heatmap showing prevalence across cuisines
- Cuisine similarity matrix based on ingredient profiles
- Charts of distinguishing ingredients for each cuisine

### Model Evaluation
- Test set: 30 recipes from each cuisine
- Overall accuracy: ~70%
- Best performance: Indian cuisine (77%)
- Most challenging: Chinese cuisine (60%)
- Detailed analysis of misclassification patterns

## 🔍 Key Findings

### Cuisine Signatures

#### 🇮🇳 Indian Cuisine
- **Key ingredients**: Cumin, coriander, garam masala
- **Distinctive element**: Yogurt usage
- **Low prevalence**: Soy sauce, fish sauce

#### 🇹🇭 Thai Cuisine
- **Key ingredients**: Fish sauce, lemongrass, coconut milk
- **Distinctive combination**: Fish sauce + cumin without yogurt
- **Flavor profile**: Sweet, sour, and spicy elements

#### 🇨🇳 Chinese Cuisine
- **Key ingredients**: Soy sauce, sesame oil, ginger
- **Notable feature**: Shares patterns with Korean cuisine
- **Distinctive**: Specific vegetable combinations

#### 🇯🇵 Japanese Cuisine
- **Key ingredients**: Mirin, sake, specific seafood
- **Notable feature**: More minimal ingredient lists
- **Flavor profile**: Prominent umami elements

#### 🇰🇷 Korean Cuisine
- **Key ingredients**: Gochujang, sesame oil, garlic
- **Distinctive**: Fermented ingredients
- **Similar to**: Chinese cuisine but with different proportions

### Practical Applications
1. **Recipe Recommendation Systems**
2. **Culinary Innovation & Fusion Development**
3. **Ingredient Substitution Guidelines**
4. **Dietary Adaptation Strategies**

## 🛠️ Installation

### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 🚀 Usage

Run the main script:

```bash
python Cuisine-Classification-using-Decision-Trees.py
```

The script will:
1. Download and prepare the dataset
2. Build and train the decision tree models
3. Evaluate model performance
4. Generate visualizations
5. Print detailed analysis and insights

## 📊 Results

### Model Performance
| Cuisine  | Accuracy | Common Misclassification |
|----------|----------|--------------------------|
| Indian   | 77%      | Thai (10%)               |
| Thai     | 73%      | Indian (13%)             |
| Japanese | 70%      | Chinese (20%)            |
| Korean   | 67%      | Chinese (23%)            |
| Chinese  | 60%      | Korean (37%)             |

![Confusion Matrix](https://raw.githubusercontent.com/EnricGE/Cuisine-Classification-using-Decision-Trees/main/results/Confusion-Matrix.png)
*Confusion Matrix Graph*

![Top Ingredients by Cuisine](https://raw.githubusercontent.com/EnricGE/Cuisine-Classification-using-Decision-Trees/main/results/Top-Ingredients-by-Cuisine.png)
*Top Ingredients by Cuisine*

![Cuisine Similarity Based on Ingredient Profiles](https://raw.githubusercontent.com/EnricGE/Cuisine-Classification-using-Decision-Trees/main/results/Cuisine-Similarity-Based-on-Ingredient-Profiles.png)
*Cuisine Similarity Based on Ingredient Profiles*

![Most Distinguishing Ingredients by Cuisine](https://raw.githubusercontent.com/EnricGE/Cuisine-Classification-using-Decision-Trees/main/results/Most-Distinguishing-Ingredients-by-Cuisine.png)
*Most Distinguishing Ingredients by Cuisine*

## 🔮 Limitations & Future Work

### Current Limitations
- Decision trees may overfit with increasing depth
- Binary ingredient representation (present/absent) lacks quantity information
- Limited to major cuisines without regional variations

### Future Improvements
- Implement ensemble methods (Random Forest, Gradient Boosting)
- Incorporate cooking methods and preparation techniques
- Expand analysis to more cuisines and regional variations
- Add quantitative ingredient measures
- Explore ingredient combinations and interactions

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built by [Enric GIL](https://github.com/EnricGE)*
