# General-Purpose Sentiment Analysis Tool for Qualitative Research
# (Works with any text: literature, reviews, social media, interviews, etc.)

# Install required packages
# pip install streamlit nltk matplotlib wordcloud textstat seaborn

# Import libraries
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import string
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
import io
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# ----------------------
# ENHANCED CONFIGURATION: Material-Specific Adaptations
# ----------------------
class AnalysisConfig:
    """
    Enhanced configuration class with material-specific presets.
    Easily switch between different text types or create custom configurations.
    """

    # Material-specific presets
    MATERIAL_PRESETS = {
        'novel': {
            'positive_threshold': 0.1,
            'negative_threshold': -0.1,
            'segment_pattern': r'chapter\s+\d+|part\s+\d+|book\s+\d+',
            'ngram_size': 3,
            'preserve_literary_devices': True,
            'context_window': 5
        },
        'news': {
            'positive_threshold': 0.05,
            'negative_threshold': -0.05,
            'segment_pattern': r'\n\n|\. [A-Z]',
            'ngram_size': 2,
            'preserve_literary_devices': False,
            'context_window': 3
        },
        'reviews': {
            'positive_threshold': 0.02,
            'negative_threshold': -0.02,
            'segment_pattern': r'pros:|cons:|rating:|score:|stars:|\/5|\/10',
            'ngram_size': 2,
            'preserve_literary_devices': False,
            'context_window': 4
        },
        'social_media': {
            'positive_threshold': 0.01,
            'negative_threshold': -0.01,
            'segment_pattern': r'#\w+|@\w+|\n',
            'ngram_size': 2,
            'preserve_literary_devices': False,
            'context_window': 3
        },
        'academic': {
            'positive_threshold': 0.15,
            'negative_threshold': -0.15,
            'segment_pattern': r'introduction|methodology|results|discussion|conclusion',
            'ngram_size': 3,
            'preserve_literary_devices': False,
            'context_window': 5
        },
        'articles': {
            'positive_threshold': 0.08,
            'negative_threshold': -0.08,
            'segment_pattern': r'\n\n|subtitle:|heading:',
            'ngram_size': 2,
            'preserve_literary_devices': False,
            'context_window': 4
        }
    }

    def __init__(self, material_type='auto'):
        """Initialize with material-specific or auto-detected settings"""
        self.material_type = material_type
        self.auto_detected_type = None

        # Base configuration
        self.BASE_STOPWORDS = set(stopwords.words('english') + list(string.punctuation))
        self.ADDITIONAL_STOPWORDS = {'could', 'would', 'might', 'must', 'said', 'says'}
        self.CUSTOM_RETAIN_WORDS = {'not', 'no', 'none', 'never', 'nothing', 'neither', 'nor'}
        self.STOPWORDS = self.BASE_STOPWORDS.union(self.ADDITIONAL_STOPWORDS) - self.CUSTOM_RETAIN_WORDS

        # Enhanced theme codebook with more comprehensive coverage
        self.THEME_CODEBOOK = {
            'emotion_positive': ['love', 'joy', 'happiness', 'delight', 'excitement', 'pleasure', 'satisfaction', 'contentment', 'bliss', 'euphoria', 'gratitude', 'hope', 'optimism', 'enthusiasm', 'admiration'],
            'emotion_negative': ['hate', 'fear', 'sadness', 'anger', 'frustration', 'disappointment', 'grief', 'despair', 'anxiety', 'worry', 'disgust', 'contempt', 'resentment', 'bitterness', 'melancholy'],
            'quality_positive': ['excellent', 'brilliant', 'outstanding', 'superb', 'magnificent', 'wonderful', 'fantastic', 'amazing', 'incredible', 'remarkable', 'impressive', 'perfect', 'superior', 'exceptional'],
            'quality_negative': ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious', 'appalling', 'deplorable', 'pathetic', 'inferior', 'mediocre', 'poor', 'bad', 'disappointing', 'unsatisfactory'],
            'power_authority': ['control', 'authority', 'power', 'dominance', 'leadership', 'influence', 'command', 'rule', 'govern', 'dictate', 'hierarchy', 'superior', 'subordinate', 'obey'],
            'conflict_tension': ['fight', 'struggle', 'conflict', 'battle', 'war', 'dispute', 'argument', 'confrontation', 'clash', 'tension', 'opposition', 'resistance', 'rebellion', 'protest'],
            'relationships': ['family', 'friend', 'partner', 'colleague', 'companion', 'ally', 'enemy', 'rival', 'neighbor', 'community', 'society', 'relationship', 'bond', 'connection'],
            'performance_arts': ['acting', 'performance', 'actor', 'character', 'role', 'scene', 'dialogue', 'script', 'plot', 'story', 'narrative', 'cinematography', 'direction', 'production'],
            'technology': ['digital', 'online', 'internet', 'computer', 'software', 'app', 'website', 'platform', 'system', 'technology', 'innovation', 'artificial', 'intelligence', 'data'],
            'business_economy': ['profit', 'loss', 'revenue', 'cost', 'price', 'market', 'business', 'economy', 'financial', 'investment', 'budget', 'expense', 'income', 'growth'],
            'health_wellness': ['health', 'medical', 'doctor', 'treatment', 'medicine', 'therapy', 'wellness', 'fitness', 'exercise', 'nutrition', 'diet', 'healing', 'recovery', 'illness'],
            'time_temporal': ['past', 'present', 'future', 'history', 'memory', 'nostalgia', 'tradition', 'modern', 'contemporary', 'ancient', 'recent', 'current', 'upcoming', 'forever']
        }

        # Apply material-specific settings
        if material_type in self.MATERIAL_PRESETS:
            self._apply_preset(material_type)
        else:
            self._set_default_values()

    def _apply_preset(self, preset_name):
        """Apply material-specific preset configuration"""
        preset = self.MATERIAL_PRESETS[preset_name]
        self.POSITIVE_THRESHOLD = preset['positive_threshold']
        self.NEGATIVE_THRESHOLD = preset['negative_threshold']
        self.SEGMENT_PATTERN = preset['segment_pattern']
        self.NGRAM_SIZE = preset['ngram_size']
        self.PRESERVE_LITERARY_DEVICES = preset['preserve_literary_devices']
        self.CONTEXT_WINDOW = preset['context_window']

    def _set_default_values(self):
        """Set default configuration values"""
        self.POSITIVE_THRESHOLD = 0.05
        self.NEGATIVE_THRESHOLD = -0.05
        self.SEGMENT_PATTERN = r'\n\n'
        self.NGRAM_SIZE = 2
        self.PRESERVE_LITERARY_DEVICES = False
        self.CONTEXT_WINDOW = 4

    def auto_detect_material_type(self, text_sample):
        """Auto-detect material type based on text characteristics"""
        text_lower = text_sample.lower()
        word_count = len(text_sample.split())

        # Detection patterns
        patterns = {
            'novel': [r'chapter\s+\d+', r'he said|she said', r'she said|he said', r'protagonist', r'character', r'\bbook\b'],
            'news': [r'\w+\s+\(reuters\)', r'breaking:', r'reported', r'according to'],
            'reviews': [r'rating:', r'stars', r'\/5', r'recommend', r'pros:', r'cons:'],
            'social_media': [r'#\w+', r'@\w+', r'lol', r'omg', r'rt\s+@', r'\blol\b', r'\bomg\b'],
            'academic': [r'abstract', r'methodology', r'hypothesis', r'references', r'et al'],
            'articles': [r'published', r'journalist', r'editor', r'source']
        }

        scores = {}
        for material_type, pattern_list in patterns.items():
            score = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in pattern_list)
            scores[material_type] = score

        # Additional heuristics - stronger weighting
        if word_count > 50000:
            scores['novel'] += 10  # Very likely a novel
        elif word_count > 10000:
            scores['novel'] += 5
        elif word_count < 500:
            scores['social_media'] += 2
        
        # Check for dialogue patterns (strong indicator of fiction)
        dialogue_count = len(re.findall(r'"[^"]{10,}"', text_sample))
        if dialogue_count > 10:
            scores['novel'] += 5

        detected_type = max(scores, key=scores.get) if max(scores.values()) > 0 else 'novel'
        self.auto_detected_type = detected_type

        return detected_type

    def update_for_detected_type(self, detected_type):
        """Update configuration based on auto-detected type"""
        if detected_type in self.MATERIAL_PRESETS:
            st.info(f"Auto-detected material type: {detected_type}")
            st.info(f"Applying optimized settings for {detected_type} analysis...")
            self._apply_preset(detected_type)
            self.material_type = detected_type
        else:
            st.info(f"Could not auto-detect material type. Using default settings.")

# ----------------------
# ENHANCED TEXT PREPROCESSING
# ----------------------
def enhanced_preprocess_text(text, config):
    """Enhanced preprocessing with material-specific adaptations"""

    # Preserve important patterns before cleaning
    preserved_elements = {}

    if config.PRESERVE_LITERARY_DEVICES:
        # Preserve literary devices for novels
        preserved_elements['emphasis'] = re.findall(r'\*[^*]+\*|_[^_]+_', text)
        preserved_elements['dialogue'] = re.findall(r'"[^"]+"', text)

    # Handle different text encodings and special characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Material-specific cleaning
    if config.material_type == 'social_media':
        # Preserve hashtags and mentions but clean URLs
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        cleaned_text = re.sub(r"[^\w\s'#@-]", ' ', text)
    elif config.material_type == 'news':
        # Remove bylines and timestamps
        text = re.sub(r'\([A-Z]+\)\s*-?', '', text)  # Remove (REUTERS), (AP) etc.
        cleaned_text = re.sub(r"[^\w\s'-]", ' ', text)
    else:
        # General cleaning
        cleaned_text = re.sub(r"[^\w\s'-]", ' ', text)

    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Calculate readability metrics
    try:
        readability_score = flesch_reading_ease(text)
        grade_level = flesch_kincaid_grade(text)
    except:
        readability_score = None
        grade_level = None

    return {
        'original': text,
        'cleaned': cleaned_text,
        'lower': cleaned_text.lower(),
        'preserved_elements': preserved_elements,
        'readability_score': readability_score,
        'grade_level': grade_level,
        'word_count': len(cleaned_text.split()),
        'sentence_count': len(sent_tokenize(text))
    }

# ----------------------
# ENHANCED SENTIMENT ANALYSIS FUNCTIONS
# ----------------------
def get_enhanced_word_sentiments(text_data, config):
    """Enhanced word-level sentiment analysis with better context handling"""
    # Limit tokens for performance - stricter for very large files
    original_text = text_data['original']
    word_count = len(original_text.split())
    
    if word_count > 100000:
        max_tokens = 5000  # Very strict for huge files
        st.warning(f"⚠️ Extremely large text detected ({word_count:,} words). Analyzing first {max_tokens:,} tokens for performance.")
    elif word_count > 50000:
        max_tokens = 7500
        st.warning(f"⚠️ Very large text ({word_count:,} words). Analyzing first {max_tokens:,} tokens for performance.")
    else:
        max_tokens = 10000
    
    tokens = word_tokenize(original_text)
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    word_details = defaultdict(lambda: {
        'score': 0, 'count': 0, 'positions': [], 'contexts': [],
        'sentence_positions': [], 'co_occurring_words': []
    })

    sentences = sent_tokenize(text_data['original'])

    for idx, token in enumerate(tokens):
        word = token.lower()

        # Enhanced stopword filtering
        if word in config.STOPWORDS and not any(neg in word for neg in ['not', 'no', 'never']):
            continue

        if not re.search(r'[a-zA-Z]', token) or len(word) < 2:
            continue

        # Calculate sentiment with negation handling
        if word_details[word]['count'] == 0:
            # Look for negations in context
            context_start = max(0, idx - 3)
            context_end = min(len(tokens), idx + 4)
            context_tokens = tokens[context_start:context_end]

            negation_words = ['not', 'no', 'never', 'neither', 'nor', 'nothing', 'nobody', 'nowhere']
            has_negation = any(neg in [t.lower() for t in context_tokens[:3]] for neg in negation_words)

            base_score = SentimentIntensityAnalyzer().polarity_scores(word)['compound']
            # Flip sentiment if negation is present
            word_details[word]['score'] = -base_score if has_negation and abs(base_score) > 0.1 else base_score

        # Track enhanced metadata
        word_details[word]['count'] += 1
        word_details[word]['positions'].append(idx + 1)

        # Find which sentence this word belongs to
        char_pos = sum(len(t) + 1 for t in tokens[:idx])
        sentence_idx = 0
        current_pos = 0
        for i, sentence in enumerate(sentences):
            if current_pos <= char_pos <= current_pos + len(sentence):
                sentence_idx = i
                break
            current_pos += len(sentence)

        word_details[word]['sentence_positions'].append(sentence_idx)

        # Capture enhanced context
        context_window = tokens[max(0, idx-config.CONTEXT_WINDOW):min(len(tokens), idx+config.CONTEXT_WINDOW+1)]
        word_details[word]['contexts'].append(' '.join(context_window))

        # Track co-occurring sentiment words
        co_occurring = [t.lower() for t in context_window if t.lower() != word and len(t) > 2]
        word_details[word]['co_occurring_words'].extend(co_occurring)

    # Format results with enhanced information
    results = []
    for word, data in word_details.items():
        if data['score'] != 0:
            results.append({
                'word': word,
                'score': data['score'],
                'count': data['count'],
                'frequency': data['count'] / len(tokens) * 100,
                'positions': data['positions'][:5],
                'sentence_spread': len(set(data['sentence_positions'])),
                'sample_contexts': list(set(data['contexts']))[:3],
                'top_co_words': [word for word, count in Counter(data['co_occurring_words']).most_common(3)]
            })

    return results

def analyze_sentiment_patterns(text_data, config):
    """Analyze sentiment patterns and trends within the text"""
    sentences = sent_tokenize(text_data['original'])
    sia = SentimentIntensityAnalyzer()

    sentence_sentiments = []
    for i, sentence in enumerate(sentences):
        scores = sia.polarity_scores(sentence)
        sentence_sentiments.append({
            'sentence_num': i + 1,
            'text': sentence[:100] + '...' if len(sentence) > 100 else sentence,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })

    # Calculate sentiment trends
    window_size = max(5, len(sentences) // 20)  # Adaptive window size
    rolling_sentiment = []

    for i in range(len(sentences) - window_size + 1):
        window_scores = [s['compound'] for s in sentence_sentiments[i:i+window_size]]
        rolling_sentiment.append({
            'position': i + window_size // 2,
            'avg_sentiment': np.mean(window_scores),
            'sentiment_variance': np.var(window_scores)
        })

    return {
        'sentence_sentiments': sentence_sentiments,
        'rolling_sentiment': rolling_sentiment,
        'overall_trend': 'increasing' if rolling_sentiment[-1]['avg_sentiment'] > rolling_sentiment[0]['avg_sentiment'] else 'decreasing',
        'sentiment_volatility': np.mean([r['sentiment_variance'] for r in rolling_sentiment])
    }

def enhanced_segment_analysis(text_data, config):
    """Enhanced segment analysis with adaptive segmentation"""
    text = text_data['lower']

    # Try multiple segmentation strategies
    segments = []

    # Strategy 1: Use configured pattern
    pattern_segments = re.split(config.SEGMENT_PATTERN, text, flags=re.IGNORECASE)
    if len(pattern_segments) > 1:
        segments = [seg.strip() for seg in pattern_segments if len(seg.strip()) > 50]

    # Strategy 2: Paragraph-based if pattern segmentation fails
    if len(segments) < 2:
        segments = [seg.strip() for seg in text.split('\n\n') if len(seg.strip()) > 50]

    # Strategy 3: Sentence-based for short texts
    if len(segments) < 2:
        sentences = sent_tokenize(text)
        segment_size = max(3, len(sentences) // 10)
        segments = []
        for i in range(0, len(sentences), segment_size):
            segment = ' '.join(sentences[i:i+segment_size])
            if len(segment.strip()) > 50:
                segments.append(segment)

    # Analyze segments
    sia = SentimentIntensityAnalyzer()
    segment_results = []

    for i, segment in enumerate(segments):
        if len(segment.strip()) < 20:
            continue

        scores = sia.polarity_scores(segment)
        segment_results.append({
            'segment': i + 1,
            'text_preview': segment[:150] + '...' if len(segment) > 150 else segment,
            'word_count': len(segment.split()),
            'sentence_count': len(sent_tokenize(segment)),
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu'],
            'sentiment_label': 'positive' if scores['compound'] > config.POSITIVE_THRESHOLD
                              else 'negative' if scores['compound'] < config.NEGATIVE_THRESHOLD
                              else 'neutral'
        })

    return segment_results

# ----------------------
# ENHANCED VISUALIZATION FUNCTIONS
# ----------------------
def create_comprehensive_visualizations(results_data, config):
    """Create a comprehensive set of visualizations"""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create subplot layout
    fig = plt.figure(figsize=(20, 16))

    # 1. Sentiment distribution pie chart
    ax1 = plt.subplot(3, 3, 1)
    sentiment_counts = results_data['sentiment_distribution']
    plt.pie([sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']],
            labels=['Positive', 'Negative', 'Neutral'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Overall Sentiment Distribution')

    # 2. Top sentiment words bar chart
    ax2 = plt.subplot(3, 3, 2)
    top_positive = results_data['top_words']['positive'][:10]
    top_negative = results_data['top_words']['negative'][:10]

    words = [w['word'] for w in top_positive + top_negative]
    scores = [w['score'] for w in top_positive + top_negative]
    colors = ['green'] * len(top_positive) + ['red'] * len(top_negative)

    plt.barh(range(len(words)), scores, color=colors, alpha=0.7)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Sentiment Score')
    plt.title('Top Sentiment Words')
    plt.grid(axis='x', alpha=0.3)

    # 3. Sentiment trends over text
    if results_data['sentiment_patterns']['rolling_sentiment']:
        ax3 = plt.subplot(3, 3, 3)
        rolling_data = results_data['sentiment_patterns']['rolling_sentiment']
        positions = [r['position'] for r in rolling_data]
        sentiments = [r['avg_sentiment'] for r in rolling_data]

        plt.plot(positions, sentiments, linewidth=2, marker='o', markersize=4)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Text Position')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Trends Throughout Text')
        plt.grid(alpha=0.3)

    # 4. Theme analysis
    ax4 = plt.subplot(3, 3, 4)
    theme_data = results_data.get('theme_analysis', {})
    if theme_data:
        themes = list(theme_data.keys())[:8]  # Top 8 themes
        theme_scores = []
        for theme in themes:
            pos_count = theme_data[theme]['positive_count']
            neg_count = theme_data[theme]['negative_count']
            total = pos_count + neg_count
            if total > 0:
                theme_scores.append((pos_count - neg_count) / total)
            else:
                theme_scores.append(0)

        colors = ['green' if score > 0 else 'red' for score in theme_scores]
        plt.barh(themes, theme_scores, color=colors, alpha=0.7)
        plt.xlabel('Net Sentiment Score')
        plt.title('Thematic Sentiment Analysis')
        plt.grid(axis='x', alpha=0.3)

    # 5. Segment analysis
    if results_data.get('segment_analysis'):
        ax5 = plt.subplot(3, 3, 5)
        segments = results_data['segment_analysis']
        segment_nums = [s['segment'] for s in segments]
        segment_scores = [s['compound_score'] for s in segments]

        colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in segment_scores]
        plt.bar(segment_nums, segment_scores, color=colors, alpha=0.7)
        plt.xlabel('Segment Number')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment by Text Segment')
        plt.grid(axis='y', alpha=0.3)

    # 6. Word frequency vs sentiment scatter
    ax6 = plt.subplot(3, 3, 6)
    all_words = results_data['word_analysis']
    frequencies = [w['frequency'] for w in all_words]
    sentiment_scores = [w['score'] for w in all_words]

    plt.scatter(frequencies, sentiment_scores, alpha=0.6, s=30)
    plt.xlabel('Word Frequency (%)')
    plt.ylabel('Sentiment Score')
    plt.title('Frequency vs Sentiment Correlation')
    plt.grid(alpha=0.3)

    # 7. Sentiment intensity distribution
    ax7 = plt.subplot(3, 3, 7)
    all_scores = [w['score'] for w in all_words]
    plt.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Sentiment Intensity Distribution')
    plt.grid(axis='y', alpha=0.3)

    # 8. Reading difficulty vs sentiment
    if results_data['text_metrics'].get('readability_score'):
        ax8 = plt.subplot(3, 3, 8)
        readability = results_data['text_metrics']['readability_score']
        overall_sentiment = results_data['overall_sentiment']['score']

        plt.scatter([readability], [overall_sentiment], s=100, color='purple', alpha=0.7)
        plt.xlabel('Reading Ease Score')
        plt.ylabel('Overall Sentiment')
        plt.title('Readability vs Sentiment')
        plt.grid(alpha=0.3)

    # 9. Material-specific insights
    ax9 = plt.subplot(3, 3, 9)
    material_type = config.material_type
    insights_text = f"Material Type: {material_type.upper()}\n\n"
    insights_text += f"Optimized for {material_type} analysis\n"
    insights_text += f"Threshold: ±{config.POSITIVE_THRESHOLD}\n"
    insights_text += f"N-gram size: {config.NGRAM_SIZE}\n"
    insights_text += f"Context window: {config.CONTEXT_WINDOW}"

    plt.text(0.1, 0.5, insights_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.axis('off')
    plt.title('Analysis Configuration')

    plt.tight_layout()
    plt.show()

def create_enhanced_wordclouds(positive_words, negative_words, theme_analysis, config):
    """Create enhanced word clouds with thematic organization"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Positive sentiment word cloud
    if positive_words:
        pos_freq = {word['word']: word['count'] for word in positive_words}
        pos_cloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='Greens', max_words=100).generate_from_frequencies(pos_freq)
        axes[0, 0].imshow(pos_cloud, interpolation='bilinear')
        axes[0, 0].set_title('Positive Sentiment Words', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

    # Negative sentiment word cloud
    if negative_words:
        neg_freq = {word['word']: word['count'] for word in negative_words}
        neg_cloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='Reds', max_words=100).generate_from_frequencies(neg_freq)
        axes[0, 1].imshow(neg_cloud, interpolation='bilinear')
        axes[0, 1].set_title('Negative Sentiment Words', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

    # Most prominent themes word cloud
    if theme_analysis:
        theme_words = {}
        for theme, data in theme_analysis.items():
            if data['total_count'] > 0:
                for word in data['key_words']:
                    theme_words[word] = theme_words.get(word, 0) + data['total_count']

        if theme_words:
            theme_cloud = WordCloud(width=800, height=400, background_color='white',
                                   colormap='viridis', max_words=100).generate_from_frequencies(theme_words)
            axes[1, 0].imshow(theme_cloud, interpolation='bilinear')
            axes[1, 0].set_title('Thematic Keywords', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')

    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = "SENTIMENT ANALYSIS SUMMARY\n\n"
    summary_text += f"✓ Positive words analyzed: {len(positive_words)}\n"
    summary_text += f"✓ Negative words analyzed: {len(negative_words)}\n"
    summary_text += f"✓ Themes identified: {len([t for t, d in theme_analysis.items() if d['total_count'] > 0])}\n"
    summary_text += f"✓ Material type: {config.material_type.upper()}\n\n"
    summary_text += "📊 This analysis provides insights into:\n"
    summary_text += "• Overall emotional tone\n"
    summary_text += "• Key sentiment drivers\n"
    summary_text += "• Thematic patterns\n"
    summary_text += "• Contextual sentiment shifts"

    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1, 1].set_title('Analysis Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ----------------------
# STREAMLIT UI AND MAIN ANALYSIS WORKFLOW
# ----------------------
def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🚀 Enhanced Sentiment Analysis Tool")
    st.markdown("**Comprehensive sentiment analysis for any text: novels, articles, reviews, news, social media, academic papers**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Material type selection
    material_type = st.sidebar.selectbox(
        "Select Material Type",
        ["auto", "novel", "news", "reviews", "social_media", "academic", "articles"],
        help="Choose the type of text for optimized analysis. 'auto' will detect automatically."
    )
    
    # Add note about large files
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: For large files (>50,000 words), analysis may take several minutes. Consider analyzing a shorter excerpt for faster results.")
    
    # File upload or text input
    st.header("📁 Input Text")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        horizontal=True
    )
    
    raw_text = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'csv', 'json'],
            help="Supported formats: .txt, .csv, .json"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    raw_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    raw_text = df.to_string()
                elif uploaded_file.type == "application/json":
                    import json
                    data = json.load(uploaded_file)
                    raw_text = str(data)
                else:
                    raw_text = str(uploaded_file.read(), "utf-8")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    else:
        raw_text = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter the text you want to analyze..."
        )
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary") and raw_text:
        if len(raw_text.strip()) < 10:
            st.warning("Please enter more text for meaningful analysis.")
            return
        
        # Warn about large files
        word_count = len(raw_text.split())
        if word_count > 10000:
            st.warning(f"⚠️ Large text detected ({word_count:,} words). Analysis may take 1-2 minutes. For faster results, consider analyzing a shorter excerpt.")
            
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize
            status_text.text("🚀 Initializing analysis...")
            progress_bar.progress(10)
            
            # Step 2: Detect material type
            status_text.text("🔍 Auto-detecting material type...")
            progress_bar.progress(20)
            
            # Step 3: Run analysis with progress
            status_text.text("📊 Analyzing sentiment (this may take a moment for large texts)...")
            progress_bar.progress(30)
            
            results_data, config = run_sentiment_analysis(raw_text, material_type)
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            display_streamlit_results(results_data, config)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during analysis: {e}")
            st.exception(e)

def run_sentiment_analysis(raw_text, material_type):
    """Run the sentiment analysis and return results"""

            # Auto-detect material type or use specified type
    config = AnalysisConfig(material_type)
    if material_type == 'auto':
        detected_type = config.auto_detect_material_type(raw_text[:5000])  # Sample first 5000 chars for better detection
        st.info(f"🔍 Auto-detected material type: **{detected_type}**. If incorrect, stop the analysis and select the correct type from the sidebar, then click 'Analyze Sentiment' again.")
        config.update_for_detected_type(detected_type)
    else:
        # User manually selected type
        config.material_type = material_type
        if material_type in config.MATERIAL_PRESETS:
            config._apply_preset(material_type)

    # Enhanced preprocessing
    processed_text = enhanced_preprocess_text(raw_text, config)

    # Run enhanced analyses
    sia = SentimentIntensityAnalyzer()
    overall_sentiment = sia.polarity_scores(processed_text['cleaned'])
    word_analysis = get_enhanced_word_sentiments(processed_text, config)
    sentiment_patterns = analyze_sentiment_patterns(processed_text, config)
    segment_analysis = enhanced_segment_analysis(processed_text, config)

    # Separate positive and negative words
    positive_words = sorted([w for w in word_analysis if w['score'] > config.POSITIVE_THRESHOLD],
                           key=lambda x: (x['score'], x['count']), reverse=True)
    negative_words = sorted([w for w in word_analysis if w['score'] < config.NEGATIVE_THRESHOLD],
                           key=lambda x: (x['score'], x['count']))

    # Theme analysis
    theme_analysis = defaultdict(lambda: {
        'positive_count': 0, 'negative_count': 0, 'total_count': 0,
        'key_words': set(), 'dominant_sentiment': 'neutral'
    })

    for word_data in word_analysis:
        word = word_data['word']
        for theme, keywords in config.THEME_CODEBOOK.items():
            if word in keywords:
                theme_analysis[theme]['total_count'] += word_data['count']
                theme_analysis[theme]['key_words'].add(word)
                if word_data['score'] > config.POSITIVE_THRESHOLD:
                    theme_analysis[theme]['positive_count'] += word_data['count']
                elif word_data['score'] < config.NEGATIVE_THRESHOLD:
                    theme_analysis[theme]['negative_count'] += word_data['count']

    # Convert sets to lists and determine dominant sentiment
    for theme, data in theme_analysis.items():
        data['key_words'] = list(data['key_words'])
        if data['positive_count'] > data['negative_count']:
            data['dominant_sentiment'] = 'positive'
        elif data['negative_count'] > data['positive_count']:
            data['dominant_sentiment'] = 'negative'

    # Calculate sentiment distribution
    total_words = len(word_analysis)
    positive_count = len(positive_words)
    negative_count = len(negative_words)
    neutral_count = total_words - positive_count - negative_count

    # Compile results
    results_data = {
        'overall_sentiment': {
            'score': overall_sentiment['compound'],
            'label': 'positive' if overall_sentiment['compound'] > config.POSITIVE_THRESHOLD
                    else 'negative' if overall_sentiment['compound'] < config.NEGATIVE_THRESHOLD
                    else 'neutral',
            'confidence': max(overall_sentiment['pos'], overall_sentiment['neg'], overall_sentiment['neu'])
        },
        'sentiment_distribution': {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        },
        'top_words': {
            'positive': positive_words[:20],
            'negative': negative_words[:20]
        },
        'word_analysis': word_analysis,
        'sentiment_patterns': sentiment_patterns,
        'segment_analysis': segment_analysis,
        'theme_analysis': dict(theme_analysis),
        'text_metrics': processed_text
    }

    return results_data, config

def display_streamlit_results(results_data, config):
    """Display results in Streamlit format"""
    
    overall = results_data['overall_sentiment']
    distribution = results_data['sentiment_distribution']
    metrics = results_data['text_metrics']
    
    # Overall sentiment display
    st.header("🎯 Overall Sentiment Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Sentiment",
            value=overall['label'].upper(),
            delta=f"{overall['score']:.3f}"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{overall['confidence']:.1%}"
        )
    
    with col3:
        st.metric(
            label="Total Words",
            value=f"{metrics['word_count']:,}"
        )
    
    with col4:
        st.metric(
            label="Sentences",
            value=f"{metrics['sentence_count']:,}"
        )
    
    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    
    col1, col2, col3 = st.columns(3)
    total_words = sum(distribution.values())
    
    with col1:
        st.metric(
            label="✅ Positive",
            value=f"{distribution['positive']} ({distribution['positive']/total_words*100:.1f}%)"
        )
    
    with col2:
        st.metric(
            label="❌ Negative", 
            value=f"{distribution['negative']} ({distribution['negative']/total_words*100:.1f}%)"
        )
    
    with col3:
        st.metric(
            label="⚪ Neutral",
            value=f"{distribution['neutral']} ({distribution['neutral']/total_words*100:.1f}%)"
        )
    
    # Text metrics
    if metrics['readability_score']:
        st.subheader("📚 Text Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ease_level = 'Very Easy' if metrics['readability_score'] > 90 else \
                        'Easy' if metrics['readability_score'] > 80 else \
                        'Fairly Easy' if metrics['readability_score'] > 70 else \
                        'Standard' if metrics['readability_score'] > 60 else \
                        'Fairly Difficult' if metrics['readability_score'] > 50 else \
                        'Difficult'
            
            st.metric(
                label="Reading Level",
                value=ease_level,
                delta=f"Score: {metrics['readability_score']:.1f}"
            )
        
        with col2:
            st.metric(
                label="Grade Level",
                value=f"{metrics['grade_level']:.1f}"
            )
    
    # Top sentiment words
    st.subheader("🔍 Key Sentiment Words")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✅ Top Positive Words**")
        if results_data['top_words']['positive']:
            pos_df = pd.DataFrame(results_data['top_words']['positive'][:10])
            st.dataframe(
                pos_df[['word', 'score', 'count', 'frequency']].round(3),
                use_container_width=True
            )
        else:
            st.write("No significant positive words found.")
    
    with col2:
        st.write("**❌ Top Negative Words**")
        if results_data['top_words']['negative']:
            neg_df = pd.DataFrame(results_data['top_words']['negative'][:10])
            st.dataframe(
                neg_df[['word', 'score', 'count', 'frequency']].round(3),
                use_container_width=True
            )
        else:
            st.write("No significant negative words found.")
    
    # Theme analysis
    st.subheader("🎭 Thematic Analysis")
    active_themes = {k: v for k, v in results_data['theme_analysis'].items() if v['total_count'] > 0}
    
    if active_themes:
        theme_data = []
        for theme, data in sorted(active_themes.items(), key=lambda x: x[1]['total_count'], reverse=True)[:10]:
            theme_data.append({
                'Theme': theme.replace('_', ' ').title(),
                'Sentiment': data['dominant_sentiment'],
                'Word Count': data['total_count'],
                'Key Terms': ', '.join(data['key_words'][:3])
            })
        
        theme_df = pd.DataFrame(theme_data)
        st.dataframe(theme_df, use_container_width=True)
    else:
        st.write("No significant thematic patterns detected.")
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    # Create and display visualizations
    create_streamlit_visualizations(results_data, config)
    
    # Methodology notes
    with st.expander("⚠️ Methodology & Limitations"):
            display_methodology_notes(config)

def create_streamlit_visualizations(results_data, config):
    """Create Streamlit-compatible visualizations"""
    
    # Sentiment distribution pie chart
    st.subheader("📊 Sentiment Distribution")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts = results_data['sentiment_distribution']
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
    colors = ['#2E8B57', '#DC143C', '#708090']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Overall Sentiment Distribution')
    st.pyplot(fig)
    
    # Top sentiment words bar chart
    st.subheader("🔍 Top Sentiment Words")
    
    top_positive = results_data['top_words']['positive'][:10]
    top_negative = results_data['top_words']['negative'][:10]
    
    if top_positive or top_negative:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        words = [w['word'] for w in top_positive + top_negative]
        scores = [w['score'] for w in top_positive + top_negative]
        colors = ['green'] * len(top_positive) + ['red'] * len(top_negative)
        
        bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Sentiment Score')
        ax.set_title('Top Sentiment Words')
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)
    
    # Sentiment trends
    if results_data['sentiment_patterns']['rolling_sentiment']:
        st.subheader("📈 Sentiment Trends")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_data = results_data['sentiment_patterns']['rolling_sentiment']
        positions = [r['position'] for r in rolling_data]
        sentiments = [r['avg_sentiment'] for r in rolling_data]
        
        ax.plot(positions, sentiments, linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Text Position')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Trends Throughout Text')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    # Theme analysis
    theme_data = results_data.get('theme_analysis', {})
    if theme_data:
        st.subheader("🎭 Thematic Sentiment Analysis")
        
        active_themes = {k: v for k, v in theme_data.items() if v['total_count'] > 0}
        if active_themes:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            themes = list(active_themes.keys())[:8]  # Top 8 themes
            theme_scores = []
            for theme in themes:
                pos_count = active_themes[theme]['positive_count']
                neg_count = active_themes[theme]['negative_count']
                total = pos_count + neg_count
                if total > 0:
                    theme_scores.append((pos_count - neg_count) / total)
                else:
                    theme_scores.append(0)
            
            colors = ['green' if score > 0 else 'red' for score in theme_scores]
            bars = ax.barh(themes, theme_scores, color=colors, alpha=0.7)
            ax.set_xlabel('Net Sentiment Score')
            ax.set_title('Thematic Sentiment Analysis')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
    
    # Word clouds
    st.subheader("☁️ Word Clouds")
    
    positive_words = results_data['top_words']['positive']
    negative_words = results_data['top_words']['negative']
    
    if positive_words or negative_words:
        col1, col2 = st.columns(2)
        
        with col1:
            if positive_words:
                st.write("**Positive Words**")
                pos_freq = {word['word']: word['count'] for word in positive_words[:50]}
                if pos_freq:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    wordcloud = WordCloud(width=400, height=300, background_color='white',
                                         colormap='Greens', max_words=50).generate_from_frequencies(pos_freq)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        
        with col2:
            if negative_words:
                st.write("**Negative Words**")
                neg_freq = {word['word']: word['count'] for word in negative_words[:50]}
                if neg_freq:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    wordcloud = WordCloud(width=400, height=300, background_color='white',
                                         colormap='Reds', max_words=50).generate_from_frequencies(neg_freq)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

def display_enhanced_results(results_data, config):
    """Display comprehensive analysis results"""

    overall = results_data['overall_sentiment']
    distribution = results_data['sentiment_distribution']

    print(f"\n🎯 OVERALL SENTIMENT ANALYSIS")
    print("=" * 60)
    print(f"📊 Overall Sentiment: {overall['label'].upper()} (Score: {overall['score']:.4f})")
    print(f"🎯 Confidence Level: {overall['confidence']:.1%}")
    print(f"📈 Sentiment Distribution:")
    print(f"   ✅ Positive: {distribution['positive']} words ({distribution['positive']/(sum(distribution.values()))*100:.1f}%)")
    print(f"   ❌ Negative: {distribution['negative']} words ({distribution['negative']/(sum(distribution.values()))*100:.1f}%)")
    print(f"   ⚪ Neutral: {distribution['neutral']} words ({distribution['neutral']/(sum(distribution.values()))*100:.1f}%)")

    # Top sentiment words with enhanced context
    print(f"\n🔍 KEY SENTIMENT INDICATORS")
    print("=" * 60)

    print(f"\n✅ TOP POSITIVE WORDS:")
    print(f"{'Word':<15} {'Score':<8} {'Count':<6} {'Spread':<7} {'Context Sample'}")
    print("-" * 80)
    for word in results_data['top_words']['positive'][:10]:
        context = word['sample_contexts'][0][:50] + '...' if word['sample_contexts'] else 'N/A'
        print(f"{word['word']:<15} {word['score']:.3f} {word['count']:<6} {word['sentence_spread']:<7} {context}")

    print(f"\n❌ TOP NEGATIVE WORDS:")
    print(f"{'Word':<15} {'Score':<8} {'Count':<6} {'Spread':<7} {'Context Sample'}")
    print("-" * 80)
    for word in results_data['top_words']['negative'][:10]:
        context = word['sample_contexts'][0][:50] + '...' if word['sample_contexts'] else 'N/A'
        print(f"{word['word']:<15} {word['score']:.3f} {word['count']:<6} {word['sentence_spread']:<7} {context}")

    # Thematic analysis
    print(f"\n🎭 THEMATIC SENTIMENT ANALYSIS")
    print("=" * 60)
    active_themes = {k: v for k, v in results_data['theme_analysis'].items() if v['total_count'] > 0}

    if active_themes:
        print(f"{'Theme':<20} {'Sentiment':<12} {'Words':<8} {'Key Terms'}")
        print("-" * 80)
        for theme, data in sorted(active_themes.items(), key=lambda x: x[1]['total_count'], reverse=True)[:10]:
            key_terms = ', '.join(data['key_words'][:3])
            print(f"{theme:<20} {data['dominant_sentiment']:<12} {data['total_count']:<8} {key_terms}")
    else:
        print("No significant thematic patterns detected.")

    # Sentiment trends
    if results_data['sentiment_patterns']['rolling_sentiment']:
        print(f"\n📈 SENTIMENT TRENDS")
        print("=" * 60)
        trend = results_data['sentiment_patterns']['overall_trend']
        volatility = results_data['sentiment_patterns']['sentiment_volatility']
        print(f"📊 Overall trend: {trend.upper()}")
        print(f"📉 Sentiment volatility: {'High' if volatility > 0.1 else 'Medium' if volatility > 0.05 else 'Low'} ({volatility:.3f})")

    # Text quality insights
    print(f"\n📚 TEXT CHARACTERISTICS")
    print("=" * 60)
    metrics = results_data['text_metrics']
    if metrics['readability_score']:
        ease_level = 'Very Easy' if metrics['readability_score'] > 90 else \
                    'Easy' if metrics['readability_score'] > 80 else \
                    'Fairly Easy' if metrics['readability_score'] > 70 else \
                    'Standard' if metrics['readability_score'] > 60 else \
                    'Fairly Difficult' if metrics['readability_score'] > 50 else \
                    'Difficult'
        print(f"📖 Reading Level: {ease_level} (Score: {metrics['readability_score']:.1f})")
        print(f"🎓 Grade Level: {metrics['grade_level']:.1f}")

    print(f"🔧 Analysis optimized for: {config.material_type.upper()}")

def display_methodology_notes(config):
    """Display enhanced methodology and limitations"""
    st.markdown("---")
    st.markdown("**Analysis Configuration:**")
    st.write(f"• Material type: {config.material_type}")
    st.write(f"• Sentiment thresholds: +{config.POSITIVE_THRESHOLD} / {config.NEGATIVE_THRESHOLD}")
    st.write(f"• Context window: {config.CONTEXT_WINDOW} words")
    st.write(f"• N-gram analysis: {config.NGRAM_SIZE}-grams")

    st.markdown("**Enhanced Features:**")
    st.write("✅ Auto-detection of material type")
    st.write("✅ Negation handling in sentiment calculation")
    st.write("✅ Enhanced context analysis")
    st.write("✅ Thematic sentiment mapping")
    st.write("✅ Sentiment trend analysis")
    st.write("✅ Reading difficulty assessment")

    st.markdown("**Important Limitations:**")
    st.write("• VADER works best with informal text; formal academic writing may need domain-specific models")
    st.write("• Sarcasm, irony, and cultural nuances may not be fully captured")
    st.write("• Context-dependent meanings (domain-specific jargon) may be misinterpreted")
    st.write("• Results should be validated with manual analysis for critical applications")

    st.markdown("**Recommendations:**")
    st.write("• Use results as a starting point for deeper qualitative analysis")
    st.write("• Adjust thresholds and themes based on your specific domain")
    st.write("• Cross-validate findings with subject matter experts")
    st.write("• Consider cultural and temporal context of the text")

# Initialize and run
if __name__ == "__main__":
    main()