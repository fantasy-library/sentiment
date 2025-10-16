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
from datetime import datetime
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
import io
import json
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

    # Calculate readability metrics (English text only)
    # Note: Flesch-Kincaid formulas are designed for English text
    # and use English syllable counting and sentence structure
    try:
        # Simple language detection heuristic
        # Check if text is primarily ASCII/English characters
        ascii_ratio = sum(c.isascii() for c in text[:1000]) / len(text[:1000]) if len(text) > 0 else 0
        
        if ascii_ratio > 0.8:  # Likely English or similar Latin-based language
            readability_score = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
            readability_note = "Based on Flesch-Kincaid (English)"
        else:
            readability_score = None
            grade_level = None
            readability_note = "Not calculated (non-English text detected)"
    except Exception as e:
        readability_score = None
        grade_level = None
        readability_note = f"Calculation failed: {str(e)}"

    return {
        'original': text,
        'cleaned': cleaned_text,
        'lower': cleaned_text.lower(),
        'preserved_elements': preserved_elements,
        'readability_score': readability_score,
        'grade_level': grade_level,
        'readability_note': readability_note,
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

            base_score = SentimentIntensityAnalyzer().polarity_scores(word)['compound']
            
            # More sophisticated negation detection
            # Only flip sentiment if negation is immediately adjacent to the word
            word_position = idx - context_start  # Position of current word in context
            has_immediate_negation = False
            
            # Check if negation word is immediately before the current word
            if word_position > 0:
                prev_token = context_tokens[word_position - 1].lower()
                if prev_token in negation_words:
                    has_immediate_negation = True
            
            # Only flip sentiment for immediate negations and strong sentiment words
            word_details[word]['score'] = -base_score if has_immediate_negation and abs(base_score) > 0.3 else base_score

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

    # Only calculate rolling sentiment if we have enough sentences
    if len(sentences) >= window_size:
        for i in range(len(sentences) - window_size + 1):
            window_scores = [s['compound'] for s in sentence_sentiments[i:i+window_size]]
            rolling_sentiment.append({
                'position': i + window_size // 2,
                'avg_sentiment': np.mean(window_scores),
                'sentiment_variance': np.var(window_scores)
            })

    # Calculate overall trend safely
    if len(rolling_sentiment) > 1:
        overall_trend = 'increasing' if rolling_sentiment[-1]['avg_sentiment'] > rolling_sentiment[0]['avg_sentiment'] else 'decreasing'
        sentiment_volatility = np.mean([r['sentiment_variance'] for r in rolling_sentiment])
    else:
        # Fallback for short texts
        overall_trend = 'stable'
        sentiment_volatility = 0.0

    return {
        'sentence_sentiments': sentence_sentiments,
        'rolling_sentiment': rolling_sentiment,
        'overall_trend': overall_trend,
        'sentiment_volatility': sentiment_volatility
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
    
    # Header with beta badge
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🚀 Enhanced Sentiment Analysis Tool")
    with col2:
        st.markdown("<div style='text-align: right; margin-top: 20px;'><span style='background-color: #FF6B6B; color: white; padding: 5px 15px; border-radius: 15px; font-weight: bold; font-size: 14px;'>BETA</span></div>", unsafe_allow_html=True)
    
    st.markdown("**Comprehensive sentiment analysis for any text: novels, articles, reviews, news, social media, academic papers, interviews, and more**")
    
    # Quick info banner
    st.info("💡 **New to this tool?** Start by uploading a text file or pasting text below, select your material type (or use auto-detect), then click 'Analyze Sentiment'. Check the 'Methodology & Limitations' section at the bottom for details.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Material type selection with descriptions
    material_type_options = {
        "auto": "🔍 Auto-detect (recommended)",
        "novel": "📚 Novel / Fiction / Literature",
        "news": "📰 News Articles / Journalism",
        "reviews": "⭐ Product Reviews / Ratings",
        "social_media": "💬 Social Media / Posts / Comments",
        "academic": "🎓 Academic Papers / Research",
        "articles": "✍️ Blog Posts / Opinion Pieces"
    }
    
    material_type_display = st.sidebar.selectbox(
        "📚 Select Material Type",
        list(material_type_options.values()),
        help="Choose the type of text for optimized sentiment thresholds and analysis. Auto-detect works well for most cases."
    )
    
    # Convert display name back to key
    material_type = [k for k, v in material_type_options.items() if v == material_type_display][0]
    
    # Show what settings are being used
    if material_type != 'auto':
        with st.sidebar.expander("ℹ️ Current Settings"):
            config_preview = AnalysisConfig(material_type)
            st.write(f"**Sentiment Thresholds:**")
            st.write(f"• Positive: ≥ {config_preview.POSITIVE_THRESHOLD}")
            st.write(f"• Negative: ≤ {config_preview.NEGATIVE_THRESHOLD}")
            st.write(f"• Context window: {config_preview.CONTEXT_WINDOW} words")
            st.write(f"• N-gram size: {config_preview.NGRAM_SIZE}")
    
    # Add note about large files
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Performance Tip**: For large files (>50,000 words), analysis may take several minutes. Consider analyzing a shorter excerpt for faster results.")
    
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
                    data = json.load(uploaded_file)
                    raw_text = str(data)
                else:
                    raw_text = str(uploaded_file.read(), "utf-8")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                raw_text = ""
    else:
        raw_text = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter the text you want to analyze..."
        )
    
    # Analyze button with better styling
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_button and raw_text:
        if len(raw_text.strip()) < 10:
            st.error("⚠️ Please enter more text for meaningful analysis (at least 10 characters).")
            st.stop()
        
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
            progress_bar.progress(100)
            status_text.empty()
            st.success("✅ Analysis complete! Scroll down to see results.")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state to preserve them
            st.session_state.analysis_results = results_data
            st.session_state.analysis_config = config
            st.session_state.raw_text = raw_text
            
            # Display results
            display_streamlit_results(results_data, config, raw_text)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during analysis: {str(e)}")
            st.write("**Troubleshooting tips:**")
            st.write("• Try a shorter text excerpt")
            st.write("• Ensure text is primarily English")
            st.write("• Check for unusual characters or formatting")
            st.write("• Try selecting a specific material type instead of auto-detect")
            with st.expander("📋 Full error details"):
                st.exception(e)
    
    # Display stored results if they exist and no new analysis is being performed
    elif 'analysis_results' in st.session_state and 'analysis_config' in st.session_state and 'raw_text' in st.session_state:
        st.markdown("---")
        st.success("📊 **Previous Analysis Results** - Results preserved for viewing and downloading")
        display_streamlit_results(st.session_state.analysis_results, st.session_state.analysis_config, st.session_state.raw_text)
    
    else:
        # Show helpful empty state
        if not raw_text:
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; padding: 40px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>👆 Upload a file or paste text above to get started</h3>
                <p style='color: #666;'>This tool will analyze sentiment, identify themes, and provide insights about your text.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show example
            with st.expander("📖 See an example"):
                st.write("**Example text types this tool can analyze:**")
                st.write("• 📚 **Novels & Fiction**: Detect emotional arcs, character sentiments, narrative tone")
                st.write("• 📰 **News Articles**: Identify bias, sentiment toward topics, journalistic tone")
                st.write("• ⭐ **Reviews**: Aggregate customer sentiment, identify pain points and praise")
                st.write("• 💬 **Social Media**: Analyze public opinion, trending sentiments, engagement")
                st.write("• 🎓 **Academic Papers**: Evaluate argumentative tone, critique patterns")
                st.write("• 🗣️ **Interviews & Transcripts**: Sentiment flow, speaker emotions, key themes")

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
    total_words = sum(w['count'] for w in word_analysis)
    positive_count = sum(w['count'] for w in positive_words)
    negative_count = sum(w['count'] for w in negative_words)
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

def run_simple_sentiment_analysis(raw_text, material_type):
    """Simplified fallback sentiment analysis"""
    config = AnalysisConfig(material_type if material_type != 'auto' else 'articles')
    
    # Basic preprocessing
    cleaned_text = re.sub(r'[^\w\s]', ' ', raw_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Basic sentiment
    sia = SentimentIntensityAnalyzer()
    overall_sentiment = sia.polarity_scores(cleaned_text)
    
    # Basic word analysis
    tokens = word_tokenize(cleaned_text.lower())
    word_scores = []
    for token in tokens[:1000]:  # Limit to first 1000 words
        if token.isalpha() and len(token) > 2:
            score = sia.polarity_scores(token)
            if score['compound'] != 0:
                word_scores.append({
                    'word': token,
                    'score': score['compound'],
                    'count': 1,
                    'frequency': 1 / len(tokens) * 100,
                    'positions': [],
                    'contexts': [],
                    'sentence_positions': [],
                    'sentence_spread': 0,
                    'sample_contexts': []
                })
    
    # Separate positive and negative
    positive_words = [w for w in word_scores if w['score'] > 0.05][:20]
    negative_words = [w for w in word_scores if w['score'] < -0.05][:20]
    
    results_data = {
        'overall_sentiment': {
            'score': overall_sentiment['compound'],
            'label': 'positive' if overall_sentiment['compound'] > 0.05
                    else 'negative' if overall_sentiment['compound'] < -0.05
                    else 'neutral',
            'confidence': max(overall_sentiment['pos'], overall_sentiment['neg'], overall_sentiment['neu'])
        },
        'sentiment_distribution': {
            'positive': sum(w['count'] for w in positive_words),
            'negative': sum(w['count'] for w in negative_words),
            'neutral': sum(w['count'] for w in word_scores) - sum(w['count'] for w in positive_words) - sum(w['count'] for w in negative_words)
        },
        'top_words': {
            'positive': positive_words,
            'negative': negative_words
        },
        'word_analysis': word_scores,
        'sentiment_patterns': {'rolling_sentiment': [], 'sentiment_changes': []},
        'segment_analysis': [],
        'theme_analysis': {},
        'text_metrics': {
            'original': raw_text,
            'cleaned': cleaned_text,
            'lower': cleaned_text.lower(),
            'preserved_elements': {},
            'readability_score': None,
            'grade_level': None,
            'readability_note': 'Not calculated in simplified mode',
            'word_count': len(tokens),
            'sentence_count': len(sent_tokenize(raw_text))
        }
    }
    
    return results_data, config

def display_streamlit_results(results_data, config, raw_text):
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
    st.subheader("📚 Text Characteristics")
    
    if metrics['readability_score']:
        col1, col2 = st.columns(2)
        
        with col1:
            ease_level = 'Very Easy' if metrics['readability_score'] > 90 else \
                        'Easy' if metrics['readability_score'] > 80 else \
                        'Fairly Easy' if metrics['readability_score'] > 70 else \
                        'Standard' if metrics['readability_score'] > 60 else \
                        'Fairly Difficult' if metrics['readability_score'] > 50 else \
                        'Difficult'
            
            st.metric(
                label="Reading Level (English)",
                value=ease_level,
                delta=f"Score: {metrics['readability_score']:.1f}",
                help="Flesch Reading Ease: 0-30=Very Difficult, 30-50=Difficult, 50-60=Fairly Difficult, 60-70=Standard, 70-80=Fairly Easy, 80-90=Easy, 90-100=Very Easy"
            )
        
        with col2:
            st.metric(
                label="U.S. Grade Level",
                value=f"{metrics['grade_level']:.1f}",
                help="Flesch-Kincaid Grade Level indicates the U.S. school grade needed to understand the text"
            )
        
        st.caption(f"ℹ️ {metrics['readability_note']}")
    else:
        st.info(f"ℹ️ Readability metrics: {metrics['readability_note']}")
        st.write("Note: Readability scores are only calculated for English text using the Flesch-Kincaid formulas.")
    
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
    
    # Research Insights & Analysis Section
    st.markdown("---")
    st.header("🎯 Research Insights & Analysis")
    
    with st.expander("💡 Key Findings & Actionable Insights - Click to expand", expanded=True):
        display_research_insights(results_data, config)
    
    # Download/Export Section
    st.markdown("---")
    st.header("💾 Download Analysis Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # HTML Report Download
        html_report = generate_analysis_report(results_data, config, raw_text)
        st.download_button(
            label="📄 Download HTML Report",
            data=html_report,
            file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            help="Download a comprehensive HTML report with all analysis results"
        )
    
    with col2:
        # JSON Report Download
        json_report = generate_json_report(results_data, config, raw_text)
        json_data = json.dumps(json_report, indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Download JSON Data",
            data=json_data,
            file_name=f"sentiment_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download raw analysis data in JSON format for further processing"
        )
    
    with col3:
        # CSV Export for word analysis
        if results_data.get('word_analysis'):
            # Create DataFrame from word analysis
            word_df = pd.DataFrame(results_data['word_analysis'])
            csv_buffer = io.StringIO()
            word_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📈 Download Word Analysis CSV",
                data=csv_data,
                file_name=f"word_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download detailed word-level sentiment analysis as CSV"
            )
    
    # Additional export options
    with st.expander("🔧 Advanced Export Options", expanded=False):
        st.markdown("**Custom Report Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_visualizations = st.checkbox("Include visualization data", value=True)
            include_raw_text = st.checkbox("Include original text", value=False)
            include_detailed_metrics = st.checkbox("Include detailed metrics", value=True)
        
        with col2:
            report_format = st.selectbox("Report Format", ["HTML", "Markdown", "Plain Text"])
            include_recommendations = st.checkbox("Include recommendations", value=True)
            include_methodology = st.checkbox("Include methodology notes", value=True)
        
        if st.button("🔄 Generate Custom Report"):
            # Generate custom report based on user preferences
            custom_report = generate_custom_report(
                results_data, config, raw_text, 
                include_visualizations, include_raw_text, include_detailed_metrics,
                include_recommendations, include_methodology, report_format
            )
            
            if report_format == "HTML":
                mime_type = "text/html"
                file_ext = "html"
            elif report_format == "Markdown":
                mime_type = "text/markdown"
                file_ext = "md"
            else:
                mime_type = "text/plain"
                file_ext = "txt"
            
            st.download_button(
                label=f"📥 Download Custom {report_format} Report",
                data=custom_report,
                file_name=f"custom_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=mime_type,
                help=f"Download custom report in {report_format} format"
            )
    
    # Methodology notes
    with st.expander("⚠️ Methodology & Limitations - Click to expand"):
            display_methodology_notes(config)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px;'>
            <p><strong>Enhanced Sentiment Analysis Tool v1.0-beta</strong></p>
            <p>Built with VADER (NLTK) • Powered by Streamlit</p>
            <p>📧 <a href="https://github.com/fantasy-library/sentiment" target="_blank">GitHub</a> • 
            <a href="https://github.com/fantasy-library/sentiment/issues" target="_blank">Report Issues</a> • 
            <a href="https://www.nltk.org/" target="_blank">NLTK Docs</a></p>
        </div>
        """, unsafe_allow_html=True)

def create_streamlit_visualizations(results_data, config):
    """Create Streamlit-compatible visualizations"""
    
    # Sentiment distribution pie chart - smaller size
    st.subheader("📊 Sentiment Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        sentiment_counts = results_data['sentiment_distribution']
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
        colors = ['#2E8B57', '#DC143C', '#708090']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('Overall Sentiment Distribution')
        st.pyplot(fig)
    
    with col2:
        # Marketing insights based on distribution
        st.markdown("**📊 Sentiment Insights:**")
        
        total = sum(sizes)
        pos_pct = (sizes[0] / total * 100) if total > 0 else 0
        neg_pct = (sizes[1] / total * 100) if total > 0 else 0
        neu_pct = (sizes[2] / total * 100) if total > 0 else 0
        
        # Sentiment balance insight
        if pos_pct > 60:
            st.success(f"✅ **Strong Positive Sentiment** ({pos_pct:.1f}%)")
            st.write("• High satisfaction or positive reception")
            st.write("• Good candidate for testimonials and case studies")
            st.write("• Leverage positive language in communications")
        elif pos_pct > 40:
            st.info(f"📊 **Balanced Positive** ({pos_pct:.1f}%)")
            st.write("• Generally favorable reception")
            st.write("• Focus on amplifying positive aspects")
            st.write("• Address neutral feedback to convert to positive")
        elif neg_pct > 40:
            st.error(f"⚠️ **High Negative Sentiment** ({neg_pct:.1f}%)")
            st.write("• Critical concerns need immediate attention")
            st.write("• Potential crisis management situation")
            st.write("• Prioritize addressing top negative themes")
        else:
            st.warning(f"⚪ **Neutral/Mixed Sentiment** (Neu: {neu_pct:.1f}%)")
            st.write("• Opportunity to strengthen messaging")
            st.write("• Focus on emotional engagement strategies")
            st.write("• Conduct deeper qualitative research")
    
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

def display_research_insights(results_data, config):
    """Display actionable research insights and analysis"""
    
    overall = results_data['overall_sentiment']
    distribution = results_data['sentiment_distribution']
    top_positive = results_data['top_words']['positive'][:10]
    top_negative = results_data['top_words']['negative'][:10]
    themes = results_data.get('theme_analysis', {})
    
    # Calculate key metrics
    total_words = sum(distribution.values())
    pos_pct = (distribution['positive'] / total_words * 100) if total_words > 0 else 0
    neg_pct = (distribution['negative'] / total_words * 100) if total_words > 0 else 0
    
    # 1. Key Sentiment Drivers Analysis
    st.subheader("🗣️ Key Sentiment Drivers Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💬 What the Text Reveals:**")
        
        # Extract top emotional drivers
        if top_positive:
            st.write("**Top Positive Elements:**")
            for i, word in enumerate(top_positive[:5], 1):
                st.write(f"{i}. **{word['word']}** (mentioned {word['count']}x) - Score: {word['score']:.2f}")
        
        if top_negative:
            st.write("\n**Top Negative Elements:**")
            for i, word in enumerate(top_negative[:5], 1):
                st.write(f"{i}. **{word['word']}** (mentioned {word['count']}x) - Score: {word['score']:.2f}")
    
    with col2:
        st.markdown("**📊 Actionable Insights:**")
        
        if pos_pct > 60:
            st.success("**Leverage Positive Elements:**")
            st.write("• Build on strengths identified in the analysis")
            st.write("• Use positive language patterns in future content")
            st.write("• Highlight successful aspects in communications")
        elif neg_pct > 30:
            st.error("**Address Negative Elements:**")
            st.write("• Focus on areas needing improvement")
            st.write("• Develop strategies to address concerns")
            st.write("• Create content that counters negative perceptions")
        else:
            st.info("**Optimize Communication:**")
            st.write("• Strengthen emotional connection in messaging")
            st.write("• Test different approaches and value propositions")
            st.write("• Develop more engaging content strategies")
    
    # 2. Overall Sentiment & Theme Analysis
    st.markdown("---")
    st.subheader("🎨 Overall Sentiment & Theme Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Overall Sentiment Assessment:**")
        
        if overall['label'] == 'positive':
            st.success(f"✅ **Positive Sentiment** (Score: {overall['score']:.2f})")
            st.write("**Key Strengths:**")
            st.write("• Strong positive emotional tone")
            st.write("• Favorable perception and reception")
            st.write("• Foundation for building on success")
        elif overall['label'] == 'negative':
            st.error(f"⚠️ **Negative Sentiment** (Score: {overall['score']:.2f})")
            st.write("**Areas of Concern:**")
            st.write("• Negative emotional tone detected")
            st.write("• Potential issues requiring attention")
            st.write("• Immediate focus needed on improvement")
        else:
            st.warning(f"⚪ **Neutral Sentiment** (Score: {overall['score']:.2f})")
            st.write("**Opportunities:**")
            st.write("• Balanced but lacks strong emotional connection")
            st.write("• Room for enhancement and differentiation")
            st.write("• Potential to develop more engaging content")
    
    with col2:
        st.markdown("**💡 Key Theme Insights:**")
        
        # Analyze themes for insights
        if themes:
            st.write("**Dominant Themes:**")
            sorted_themes = sorted(themes.items(), 
                                 key=lambda x: x[1]['total_count'], 
                                 reverse=True)[:3]
            
            for theme, data in sorted_themes:
                sentiment_label = data.get('dominant_sentiment', 'neutral')
                emoji = "✅" if sentiment_label == 'positive' else "❌" if sentiment_label == 'negative' else "⚪"
                st.write(f"{emoji} **{theme.title()}**: {data['total_count']} mentions ({sentiment_label})")
        
        st.write("\n**Strategic Recommendations:**")
        if pos_pct > 60:
            st.write("• Build on positive themes and strengths")
            st.write("• Emphasize quality and reliability aspects")
        elif neg_pct > 40:
            st.write("• Focus on addressing negative themes")
            st.write("• Develop strategies to improve perception")
        else:
            st.write("• Develop unique value propositions")
            st.write("• Create more emotionally engaging content")
    
    # 3. Communication Strategy & Language Insights
    st.markdown("---")
    st.subheader("✍️ Communication Strategy & Language Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Content & Communication Opportunities:**")
        
        if config.material_type == 'reviews':
            st.write("**Review Analysis Insights:**")
            st.write("• Highlight positive experiences and success stories")
            st.write("• Address common concerns and questions")
            st.write("• Develop comparison guides and educational content")
        elif config.material_type == 'social_media':
            st.write("**Social Media Intelligence:**")
            st.write("• Create engagement around positive themes")
            st.write("• Develop response strategies for negative sentiment")
            st.write("• Identify collaboration and partnership opportunities")
        elif config.material_type == 'news':
            st.write("**Media & News Analysis:**")
            st.write("• Develop communication strategy for negative coverage")
            st.write("• Leverage positive story angles and narratives")
            st.write("• Create thought leadership and expert content")
        elif config.material_type == 'academic':
            st.write("**Academic & Research Analysis:**")
            st.write("• Highlight key findings and contributions")
            st.write("• Address methodological concerns or limitations")
            st.write("• Develop follow-up research directions")
        else:
            st.write("**General Communication Opportunities:**")
            st.write("• Feature success stories and positive outcomes")
            st.write("• Create educational content addressing concerns")
            st.write("• Develop compelling narratives and storytelling")
    
    with col2:
        st.markdown("**🎯 Language & Messaging Framework:**")
        
        # Extract messaging keywords
        if top_positive and top_negative:
            st.write("**Effective Words to Emphasize:**")
            power_words = [w['word'] for w in top_positive[:5]]
            st.write(", ".join(power_words))
            
            st.write("\n**Words to Address or Reframe:**")
            avoid_words = [w['word'] for w in top_negative[:5]]
            st.write(", ".join(avoid_words))
            
            st.write("\n**Communication Priority:**")
            if pos_pct > neg_pct * 2:
                st.write("1. Amplify positive attributes and strengths")
                st.write("2. Maintain consistent positive messaging")
                st.write("3. Monitor for emerging issues or concerns")
            elif neg_pct > pos_pct:
                st.write("1. Address negative perceptions and concerns first")
                st.write("2. Rebuild trust through transparency and action")
                st.write("3. Gradually introduce positive messaging")
            else:
                st.write("1. Balance problem-solution messaging")
                st.write("2. Differentiate through unique value propositions")
                st.write("3. Build stronger emotional connections")
    
    # 4. Sentiment Trends & Audience Insights
    st.markdown("---")
    st.subheader("👥 Sentiment Trends & Audience Insights")
    
    sentiment_patterns = results_data.get('sentiment_patterns', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Sentiment Trends:**")
        
        if sentiment_patterns.get('rolling_sentiment'):
            rolling = sentiment_patterns['rolling_sentiment']
            # Extract avg_sentiment values from the rolling sentiment data
            avg_sentiments = [r['avg_sentiment'] for r in rolling]
            avg_sentiment = sum(avg_sentiments) / len(avg_sentiments) if avg_sentiments else 0
            
            if len(rolling) > 1:
                trend = "improving" if rolling[-1]['avg_sentiment'] > rolling[0]['avg_sentiment'] else "declining" if rolling[-1]['avg_sentiment'] < rolling[0]['avg_sentiment'] else "stable"
                st.write(f"**Sentiment Trend:** {trend.title()}")
                
                if trend == "improving":
                    st.success("• Positive momentum detected - build on this success!")
                    st.write("• Expand successful initiatives and approaches")
                    st.write("• Increase investment in positive areas")
                elif trend == "declining":
                    st.error("• Negative trend detected - investigate root causes")
                    st.write("• Conduct immediate analysis of concerns")
                    st.write("• Implement improvement strategies")
        
        st.write("\n**Audience Segments:**")
        if pos_pct > 60:
            st.write("🟢 **Positive Advocates**: Leverage for testimonials and referrals")
            st.write("🟡 **Satisfied Audience**: Opportunities for deeper engagement")
        if neg_pct > 20:
            st.write("🔴 **Concerned Audience**: Focus on addressing issues and rebuilding trust")
    
    with col2:
        st.markdown("**🎯 Engagement Strategy:**")
        
        st.write("**Strategic Recommendations:**")
        
        if config.material_type == 'reviews':
            st.write("• **Feedback Campaign**: Request detailed reviews from satisfied users")
            st.write("• **Issue Resolution**: Address concerns of neutral reviewers")
            st.write("• **Advocacy Program**: Reward and amplify positive voices")
        elif config.material_type == 'social_media':
            st.write("• **Influencer Collaboration**: Partner with positive voices")
            st.write("• **Community Engagement**: Build relationships with active participants")
            st.write("• **Content Strategy**: Target audiences with similar positive sentiment")
        elif config.material_type == 'academic':
            st.write("• **Research Collaboration**: Partner with positive contributors")
            st.write("• **Knowledge Sharing**: Engage with interested academic communities")
            st.write("• **Publication Strategy**: Highlight positive findings and contributions")
        else:
            st.write("• **Personalization**: Tailor messaging by sentiment group")
            st.write("• **A/B Testing**: Test different approaches and appeals")
            st.write("• **Re-engagement**: Focus on neutral/negative audience segments")
        
        st.write("\n**Quick Win Actions:**")
        st.write("1. Create compelling content for neutral audiences")
        st.write("2. Showcase positive feedback and success stories")
        st.write("3. Develop resources addressing top concerns")
    
    # 5. Strategic Insights & Next Steps
    st.markdown("---")
    st.subheader("🔍 Strategic Insights & Next Steps")
    
    st.markdown("**💼 Key Strategic Insights:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Opportunities & Strengths:**")
        
        if pos_pct > 50:
            st.write("• **Strength**: Leverage positive sentiment as a key differentiator")
            st.write("• **Strategy**: Build on successful elements and positive themes")
            st.write("• **Application**: Emphasize strengths in communications and positioning")
        
        if themes:
            st.write("\n**Theme Analysis:**")
            neutral_themes = [t for t, d in themes.items() if d.get('dominant_sentiment') == 'neutral']
            if neutral_themes:
                st.write(f"• Explore opportunities around: {', '.join(neutral_themes[:3])}")
    
    with col2:
        st.markdown("**Areas for Improvement:**")
        
        if neg_pct > 30:
            st.write("• **Alert**: Negative sentiment requires immediate attention")
            st.write("• **Action**: Develop targeted strategies to address concerns")
            st.write("• **Monitor**: Track sentiment changes and improvement progress")
        
        st.write("\n**Recommended Next Steps:**")
        st.write("1. Conduct deeper analysis of key themes identified")
        st.write("2. Gather additional feedback on areas of concern")
        st.write("3. Test different approaches based on insights")
        st.write("4. Establish ongoing sentiment monitoring")

def generate_analysis_report(results_data, config, raw_text):
    """Generate a comprehensive analysis report"""
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract key data
    overall = results_data['overall_sentiment']
    distribution = results_data['sentiment_distribution']
    top_positive = results_data['top_words']['positive'][:10]
    top_negative = results_data['top_words']['negative'][:10]
    themes = results_data.get('theme_analysis', {})
    sentiment_patterns = results_data.get('sentiment_patterns', {})
    text_metrics = results_data.get('text_metrics', {})
    
    # Calculate percentages
    total_words = sum(distribution.values())
    pos_pct = (distribution['positive'] / total_words * 100) if total_words > 0 else 0
    neg_pct = (distribution['negative'] / total_words * 100) if total_words > 0 else 0
    neu_pct = (distribution['neutral'] / total_words * 100) if total_words > 0 else 0
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .neutral {{ color: #6c757d; }}
            .word-list {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Sentiment Analysis Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Material Type:</strong> {config.material_type.title()}</p>
            <p><strong>Text Length:</strong> {len(raw_text):,} characters, {len(raw_text.split()):,} words</p>
        </div>
        
        <div class="summary">
            <h2>📋 Executive Summary</h2>
            <p><strong>Overall Sentiment:</strong> <span class="{'positive' if overall['label'] == 'positive' else 'negative' if overall['label'] == 'negative' else 'neutral'}">{overall['label'].title()}</span> (Score: {overall['score']:.3f})</p>
            <p><strong>Confidence Level:</strong> {overall['confidence']:.1%}</p>
            <p><strong>Sentiment Distribution:</strong> {pos_pct:.1f}% Positive, {neg_pct:.1f}% Negative, {neu_pct:.1f}% Neutral</p>
        </div>
        
        <div class="section">
            <h2>📈 Detailed Sentiment Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Percentage</th></tr>
                <tr><td>Positive Words</td><td>{distribution['positive']:,}</td><td class="positive">{pos_pct:.1f}%</td></tr>
                <tr><td>Negative Words</td><td>{distribution['negative']:,}</td><td class="negative">{neg_pct:.1f}%</td></tr>
                <tr><td>Neutral Words</td><td>{distribution['neutral']:,}</td><td class="neutral">{neu_pct:.1f}%</td></tr>
                <tr><td>Total Words Analyzed</td><td>{total_words:,}</td><td>100.0%</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🔝 Top Sentiment Words</h2>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h3 class="positive">Top Positive Words</h3>
                    <div class="word-list">
    """
    
    for i, word in enumerate(top_positive, 1):
        html_report += f"<p>{i}. <strong>{word['word']}</strong> (Score: {word['score']:.3f}, Count: {word['count']})</p>"
    
    html_report += """
                    </div>
                </div>
                <div style="flex: 1;">
                    <h3 class="negative">Top Negative Words</h3>
                    <div class="word-list">
    """
    
    for i, word in enumerate(top_negative, 1):
        html_report += f"<p>{i}. <strong>{word['word']}</strong> (Score: {word['score']:.3f}, Count: {word['count']})</p>"
    
    html_report += """
                    </div>
                </div>
            </div>
        </div>
    """
    
    # Add theme analysis if available
    if themes:
        html_report += """
        <div class="section">
            <h2>🎨 Theme Analysis</h2>
            <table>
                <tr><th>Theme</th><th>Total Mentions</th><th>Positive</th><th>Negative</th><th>Dominant Sentiment</th></tr>
        """
        
        for theme, data in themes.items():
            sentiment_label = data.get('dominant_sentiment', 'neutral')
            sentiment_class = 'positive' if sentiment_label == 'positive' else 'negative' if sentiment_label == 'negative' else 'neutral'
            html_report += f"""
                <tr>
                    <td><strong>{theme.title()}</strong></td>
                    <td>{data['total_count']}</td>
                    <td class="positive">{data['positive_count']}</td>
                    <td class="negative">{data['negative_count']}</td>
                    <td class="{sentiment_class}">{sentiment_label.title()}</td>
                </tr>
            """
        
        html_report += "</table></div>"
    
    # Add sentiment patterns if available
    if sentiment_patterns.get('rolling_sentiment'):
        html_report += """
        <div class="section">
            <h2>📊 Sentiment Trends</h2>
        """
        
        if len(sentiment_patterns['rolling_sentiment']) > 1:
            trend = sentiment_patterns.get('overall_trend', 'stable')
            volatility = sentiment_patterns.get('sentiment_volatility', 0)
            html_report += f"""
            <div class="metric">
                <p><strong>Overall Trend:</strong> {trend.title()}</p>
                <p><strong>Sentiment Volatility:</strong> {volatility:.3f}</p>
            </div>
            """
        
        html_report += "</div>"
    
    # Add text metrics if available
    if text_metrics:
        html_report += """
        <div class="section">
            <h2>📝 Text Quality Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        if 'readability_score' in text_metrics:
            html_report += f"<tr><td>Readability Score</td><td>{text_metrics['readability_score']:.1f}</td></tr>"
        if 'grade_level' in text_metrics:
            html_report += f"<tr><td>Grade Level</td><td>{text_metrics['grade_level']:.1f}</td></tr>"
        if 'sentence_count' in text_metrics:
            html_report += f"<tr><td>Sentence Count</td><td>{text_metrics['sentence_count']:,}</td></tr>"
        if 'avg_sentence_length' in text_metrics:
            html_report += f"<tr><td>Average Sentence Length</td><td>{text_metrics['avg_sentence_length']:.1f} words</td></tr>"
        
        html_report += "</table></div>"
    
    # Add recommendations
    html_report += """
        <div class="section">
            <h2>💡 Key Insights & Recommendations</h2>
            <div class="metric">
    """
    
    if pos_pct > 60:
        html_report += """
                <h3 class="positive">Strengths to Leverage</h3>
                <ul>
                    <li>Strong positive sentiment - build on this success</li>
                    <li>Use positive language patterns in future content</li>
                    <li>Highlight successful aspects in communications</li>
                </ul>
        """
    elif neg_pct > 30:
        html_report += """
                <h3 class="negative">Areas for Improvement</h3>
                <ul>
                    <li>Focus on areas needing improvement</li>
                    <li>Develop strategies to address concerns</li>
                    <li>Create content that counters negative perceptions</li>
                </ul>
        """
    else:
        html_report += """
                <h3 class="neutral">Optimization Opportunities</h3>
                <ul>
                    <li>Strengthen emotional connection in messaging</li>
                    <li>Test different approaches and value propositions</li>
                    <li>Develop more engaging content strategies</li>
                </ul>
        """
    
    html_report += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Methodology</h2>
            <div class="metric">
                <p><strong>Analysis Engine:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)</p>
                <p><strong>Material Type:</strong> {}</p>
                <p><strong>Positive Threshold:</strong> {}</p>
                <p><strong>Negative Threshold:</strong> {}</p>
                <p><strong>Analysis Date:</strong> {}</p>
            </div>
        </div>
    """.format(
        config.material_type.title(),
        config.POSITIVE_THRESHOLD,
        config.NEGATIVE_THRESHOLD,
        timestamp
    )
    
    html_report += """
        <footer style="margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
            <p>Report generated by General-Purpose Sentiment Analysis Tool</p>
            <p>For questions about this analysis, please refer to the methodology section above.</p>
        </footer>
    </body>
    </html>
    """
    
    return html_report

def generate_json_report(results_data, config, raw_text):
    """Generate a JSON report for programmatic access"""
    
    # Create comprehensive JSON report
    json_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "material_type": config.material_type,
            "text_length_chars": len(raw_text),
            "text_length_words": len(raw_text.split()),
            "analysis_config": {
                "positive_threshold": config.POSITIVE_THRESHOLD,
                "negative_threshold": config.NEGATIVE_THRESHOLD,
                "context_window": config.CONTEXT_WINDOW
            }
        },
        "overall_sentiment": results_data['overall_sentiment'],
        "sentiment_distribution": results_data['sentiment_distribution'],
        "top_words": {
            "positive": results_data['top_words']['positive'][:20],
            "negative": results_data['top_words']['negative'][:20]
        },
        "theme_analysis": results_data.get('theme_analysis', {}),
        "sentiment_patterns": results_data.get('sentiment_patterns', {}),
        "text_metrics": results_data.get('text_metrics', {}),
        "word_analysis": results_data.get('word_analysis', [])[:100]  # Limit for file size
    }
    
    return json_report

def generate_custom_report(results_data, config, raw_text, include_visualizations, include_raw_text, 
                          include_detailed_metrics, include_recommendations, include_methodology, report_format):
    """Generate a custom report based on user preferences"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract key data
    overall = results_data['overall_sentiment']
    distribution = results_data['sentiment_distribution']
    top_positive = results_data['top_words']['positive'][:10]
    top_negative = results_data['top_words']['negative'][:10]
    themes = results_data.get('theme_analysis', {})
    sentiment_patterns = results_data.get('sentiment_patterns', {})
    text_metrics = results_data.get('text_metrics', {})
    
    # Calculate percentages
    total_words = sum(distribution.values())
    pos_pct = (distribution['positive'] / total_words * 100) if total_words > 0 else 0
    neg_pct = (distribution['negative'] / total_words * 100) if total_words > 0 else 0
    neu_pct = (distribution['neutral'] / total_words * 100) if total_words > 0 else 0
    
    if report_format == "HTML":
        return generate_html_custom_report(
            overall, distribution, top_positive, top_negative, themes, 
            sentiment_patterns, text_metrics, config, raw_text, timestamp,
            include_visualizations, include_raw_text, include_detailed_metrics,
            include_recommendations, include_methodology
        )
    elif report_format == "Markdown":
        return generate_markdown_custom_report(
            overall, distribution, top_positive, top_negative, themes,
            sentiment_patterns, text_metrics, config, raw_text, timestamp,
            include_visualizations, include_raw_text, include_detailed_metrics,
            include_recommendations, include_methodology
        )
    else:  # Plain Text
        return generate_text_custom_report(
            overall, distribution, top_positive, top_negative, themes,
            sentiment_patterns, text_metrics, config, raw_text, timestamp,
            include_visualizations, include_raw_text, include_detailed_metrics,
            include_recommendations, include_methodology
        )

def generate_html_custom_report(overall, distribution, top_positive, top_negative, themes,
                               sentiment_patterns, text_metrics, config, raw_text, timestamp,
                               include_visualizations, include_raw_text, include_detailed_metrics,
                               include_recommendations, include_methodology):
    """Generate custom HTML report"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Custom Sentiment Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .neutral {{ color: #6c757d; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Custom Sentiment Analysis Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Material Type:</strong> {config.material_type.title()}</p>
        </div>
        
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <p><strong>Overall Sentiment:</strong> <span class="{'positive' if overall['label'] == 'positive' else 'negative' if overall['label'] == 'negative' else 'neutral'}">{overall['label'].title()}</span> (Score: {overall['score']:.3f})</p>
            <p><strong>Confidence Level:</strong> {overall['confidence']:.1%}</p>
        </div>
    """
    
    if include_detailed_metrics:
        html += f"""
        <div class="section">
            <h2>📈 Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Percentage</th></tr>
                <tr><td>Positive Words</td><td>{distribution['positive']:,}</td><td class="positive">{pos_pct:.1f}%</td></tr>
                <tr><td>Negative Words</td><td>{distribution['negative']:,}</td><td class="negative">{neg_pct:.1f}%</td></tr>
                <tr><td>Neutral Words</td><td>{distribution['neutral']:,}</td><td class="neutral">{neu_pct:.1f}%</td></tr>
            </table>
        </div>
        """
    
    if top_positive or top_negative:
        html += """
        <div class="section">
            <h2>🔝 Top Sentiment Words</h2>
        """
        if top_positive:
            html += "<h3 class='positive'>Top Positive Words</h3><ul>"
            for word in top_positive:
                html += f"<li><strong>{word['word']}</strong> (Score: {word['score']:.3f}, Count: {word['count']})</li>"
            html += "</ul>"
        
        if top_negative:
            html += "<h3 class='negative'>Top Negative Words</h3><ul>"
            for word in top_negative:
                html += f"<li><strong>{word['word']}</strong> (Score: {word['score']:.3f}, Count: {word['count']})</li>"
            html += "</ul>"
        
        html += "</div>"
    
    if include_raw_text:
        html += f"""
        <div class="section">
            <h2>📝 Original Text</h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;">
                <pre>{raw_text[:5000]}{'...' if len(raw_text) > 5000 else ''}</pre>
            </div>
        </div>
        """
    
    if include_recommendations:
        html += """
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="metric">
        """
        
        if pos_pct > 60:
            html += """
                <h3 class="positive">Strengths to Leverage</h3>
                <ul>
                    <li>Strong positive sentiment - build on this success</li>
                    <li>Use positive language patterns in future content</li>
                    <li>Highlight successful aspects in communications</li>
                </ul>
            """
        elif neg_pct > 30:
            html += """
                <h3 class="negative">Areas for Improvement</h3>
                <ul>
                    <li>Focus on areas needing improvement</li>
                    <li>Develop strategies to address concerns</li>
                    <li>Create content that counters negative perceptions</li>
                </ul>
            """
        else:
            html += """
                <h3 class="neutral">Optimization Opportunities</h3>
                <ul>
                    <li>Strengthen emotional connection in messaging</li>
                    <li>Test different approaches and value propositions</li>
                    <li>Develop more engaging content strategies</li>
                </ul>
            """
        
        html += "</div></div>"
    
    if include_methodology:
        html += f"""
        <div class="section">
            <h2>🔬 Methodology</h2>
            <div class="metric">
                <p><strong>Analysis Engine:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)</p>
                <p><strong>Material Type:</strong> {config.material_type.title()}</p>
                <p><strong>Positive Threshold:</strong> {config.POSITIVE_THRESHOLD}</p>
                <p><strong>Negative Threshold:</strong> {config.NEGATIVE_THRESHOLD}</p>
                <p><strong>Analysis Date:</strong> {timestamp}</p>
            </div>
        </div>
        """
    
    html += """
        <footer style="margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
            <p>Custom report generated by General-Purpose Sentiment Analysis Tool</p>
        </footer>
    </body>
    </html>
    """
    
    return html

def generate_markdown_custom_report(overall, distribution, top_positive, top_negative, themes,
                                   sentiment_patterns, text_metrics, config, raw_text, timestamp,
                                   include_visualizations, include_raw_text, include_detailed_metrics,
                                   include_recommendations, include_methodology):
    """Generate custom Markdown report"""
    
    md = f"""# 📊 Custom Sentiment Analysis Report

**Generated:** {timestamp}  
**Material Type:** {config.material_type.title()}

## 📋 Executive Summary

- **Overall Sentiment:** {overall['label'].title()} (Score: {overall['score']:.3f})
- **Confidence Level:** {overall['confidence']:.1%}
"""
    
    if include_detailed_metrics:
        pos_pct = (distribution['positive'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neg_pct = (distribution['negative'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neu_pct = (distribution['neutral'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        
        md += f"""
## 📈 Detailed Metrics

| Metric | Value | Percentage |
|--------|-------|------------|
| Positive Words | {distribution['positive']:,} | {pos_pct:.1f}% |
| Negative Words | {distribution['negative']:,} | {neg_pct:.1f}% |
| Neutral Words | {distribution['neutral']:,} | {neu_pct:.1f}% |
"""
    
    if top_positive or top_negative:
        md += "\n## 🔝 Top Sentiment Words\n"
        
        if top_positive:
            md += "\n### Top Positive Words\n"
            for i, word in enumerate(top_positive, 1):
                md += f"{i}. **{word['word']}** (Score: {word['score']:.3f}, Count: {word['count']})\n"
        
        if top_negative:
            md += "\n### Top Negative Words\n"
            for i, word in enumerate(top_negative, 1):
                md += f"{i}. **{word['word']}** (Score: {word['score']:.3f}, Count: {word['count']})\n"
    
    if include_raw_text:
        md += f"\n## 📝 Original Text\n\n```\n{raw_text[:2000]}{'...' if len(raw_text) > 2000 else ''}\n```\n"
    
    if include_recommendations:
        pos_pct = (distribution['positive'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neg_pct = (distribution['negative'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        
        md += "\n## 💡 Recommendations\n"
        
        if pos_pct > 60:
            md += """
### Strengths to Leverage
- Strong positive sentiment - build on this success
- Use positive language patterns in future content
- Highlight successful aspects in communications
"""
        elif neg_pct > 30:
            md += """
### Areas for Improvement
- Focus on areas needing improvement
- Develop strategies to address concerns
- Create content that counters negative perceptions
"""
        else:
            md += """
### Optimization Opportunities
- Strengthen emotional connection in messaging
- Test different approaches and value propositions
- Develop more engaging content strategies
"""
    
    if include_methodology:
        md += f"""
## 🔬 Methodology

- **Analysis Engine:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Material Type:** {config.material_type.title()}
- **Positive Threshold:** {config.POSITIVE_THRESHOLD}
- **Negative Threshold:** {config.NEGATIVE_THRESHOLD}
- **Analysis Date:** {timestamp}

---
*Custom report generated by General-Purpose Sentiment Analysis Tool*
"""
    
    return md

def generate_text_custom_report(overall, distribution, top_positive, top_negative, themes,
                               sentiment_patterns, text_metrics, config, raw_text, timestamp,
                               include_visualizations, include_raw_text, include_detailed_metrics,
                               include_recommendations, include_methodology):
    """Generate custom plain text report"""
    
    text = f"""CUSTOM SENTIMENT ANALYSIS REPORT
{'='*50}

Generated: {timestamp}
Material Type: {config.material_type.title()}

EXECUTIVE SUMMARY
{'-'*20}
Overall Sentiment: {overall['label'].title()} (Score: {overall['score']:.3f})
Confidence Level: {overall['confidence']:.1%}
"""
    
    if include_detailed_metrics:
        pos_pct = (distribution['positive'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neg_pct = (distribution['negative'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neu_pct = (distribution['neutral'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        
        text += f"""
DETAILED METRICS
{'-'*20}
Positive Words: {distribution['positive']:,} ({pos_pct:.1f}%)
Negative Words: {distribution['negative']:,} ({neg_pct:.1f}%)
Neutral Words: {distribution['neutral']:,} ({neu_pct:.1f}%)
"""
    
    if top_positive or top_negative:
        text += "\nTOP SENTIMENT WORDS\n" + "-"*20 + "\n"
        
        if top_positive:
            text += "\nTop Positive Words:\n"
            for i, word in enumerate(top_positive, 1):
                text += f"{i}. {word['word']} (Score: {word['score']:.3f}, Count: {word['count']})\n"
        
        if top_negative:
            text += "\nTop Negative Words:\n"
            for i, word in enumerate(top_negative, 1):
                text += f"{i}. {word['word']} (Score: {word['score']:.3f}, Count: {word['count']})\n"
    
    if include_raw_text:
        text += f"\nORIGINAL TEXT\n{'-'*20}\n{raw_text[:1000]}{'...' if len(raw_text) > 1000 else ''}\n"
    
    if include_recommendations:
        pos_pct = (distribution['positive'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        neg_pct = (distribution['negative'] / sum(distribution.values()) * 100) if sum(distribution.values()) > 0 else 0
        
        text += "\nRECOMMENDATIONS\n" + "-"*20 + "\n"
        
        if pos_pct > 60:
            text += """
Strengths to Leverage:
- Strong positive sentiment - build on this success
- Use positive language patterns in future content
- Highlight successful aspects in communications
"""
        elif neg_pct > 30:
            text += """
Areas for Improvement:
- Focus on areas needing improvement
- Develop strategies to address concerns
- Create content that counters negative perceptions
"""
        else:
            text += """
Optimization Opportunities:
- Strengthen emotional connection in messaging
- Test different approaches and value propositions
- Develop more engaging content strategies
"""
    
    if include_methodology:
        text += f"""
METHODOLOGY
{'-'*20}
Analysis Engine: VADER (Valence Aware Dictionary and sEntiment Reasoner)
Material Type: {config.material_type.title()}
Positive Threshold: {config.POSITIVE_THRESHOLD}
Negative Threshold: {config.NEGATIVE_THRESHOLD}
Analysis Date: {timestamp}

Custom report generated by General-Purpose Sentiment Analysis Tool
"""
    
    return text

def display_methodology_notes(config):
    """Display enhanced methodology and limitations"""
    st.markdown("---")
    
    st.markdown("### 🔬 Models & Tools Used")
    st.markdown("""
    **Sentiment Analysis:**
    - **VADER (Valence Aware Dictionary and sEntiment Reasoner)** - NLTK implementation
      - **Core**: Lexicon-based sentiment analysis with 7,500+ lexical features
      - **Originally optimized for**: Social media and informal text
      - **Extended in this tool for**: Novels, news articles, academic papers, reviews, interviews, and general text
      - **Special handling for**:
        - Emojis and emoticons (e.g., 😊, :), :(  )
        - Capitalization emphasis (e.g., "AMAZING" vs "amazing")
        - Punctuation emphasis (e.g., "good!!!" vs "good")
        - Negations (e.g., "not good", "never bad")
        - Degree modifiers (e.g., "very good", "extremely bad")
        - Contrasting conjunctions (e.g., "but", "however")
      - **Material-specific adaptations**:
        - Adjustable sentiment thresholds for different text types
        - Context window analysis (3-5 words) for nuanced understanding
        - N-gram analysis (2-3 word phrases) for idiomatic expressions
        - Literary device preservation for narrative fiction
      - **Version**: NLTK 3.x VADER Lexicon
    
    **Readability Analysis:**
    - **Flesch Reading Ease** - Textstat library
      - Formula: 206.835 - 1.015(total words/total sentences) - 84.6(total syllables/total words)
      - Score range: 0-100 (higher = easier to read)
      - **Designed for English text only**
    - **Flesch-Kincaid Grade Level** - Textstat library
      - Indicates U.S. school grade level required to understand text
      - Based on sentence length and syllable counts
      - **Calibrated for native English speakers**
    
    **Text Processing:**
    - **NLTK Tokenizers** - Word and sentence tokenization
    - **Python regex (re)** - Pattern matching and text cleaning
    - **Python Collections** - Data aggregation (defaultdict, Counter)
    """)
    
    st.markdown("### ⚙️ Analysis Configuration")
    st.write(f"• **Material type:** {config.material_type}")
    st.write(f"• **Sentiment thresholds:** Positive ≥ +{config.POSITIVE_THRESHOLD}, Negative ≤ {config.NEGATIVE_THRESHOLD}")
    st.write(f"• **Context window:** {config.CONTEXT_WINDOW} words")
    st.write(f"• **N-gram analysis:** {config.NGRAM_SIZE}-grams")

    st.markdown("### ✨ Enhanced Features")
    st.write("✅ Auto-detection of material type (novel, news, reviews, social media, academic, articles)")
    st.write("✅ Negation handling in sentiment calculation (e.g., 'not good' → negative)")
    st.write("✅ Enhanced context analysis with surrounding words")
    st.write("✅ Thematic sentiment mapping across predefined themes")
    st.write("✅ Sentiment trend analysis across text segments")
    st.write("✅ Reading difficulty assessment (English only)")

    st.markdown("### ⚠️ Important Limitations")
    
    st.markdown("**Language Support:**")
    st.write("• **Sentiment analysis:** Primarily designed for **English text**")
    st.write("  - VADER lexicon contains English words and slang")
    st.write("  - May produce unreliable results for non-English languages")
    st.write("  - Recommendation: Use language-specific sentiment models for other languages")
    st.write("• **Readability metrics:** **English only** (Flesch-Kincaid formulas)")
    st.write("  - Based on English syllable patterns and sentence structures")
    st.write("  - Not applicable to other languages (will show 'Not calculated')")
    
    st.markdown("**Model Limitations:**")
    st.write("• **VADER** works best with informal, modern text (social media, reviews, news)")
    st.write("  - May underperform on: formal academic writing, historical texts, specialized jargon")
    st.write("  - Does not fully capture: sarcasm, irony, complex metaphors, cultural context")
    st.write("• **Lexicon-based approach** has no machine learning or context understanding")
    st.write("  - Sentiment determined by word lookup, not semantic meaning")
    st.write("  - Cannot understand domain-specific meanings (e.g., 'positive test result' in medicine)")
    
    st.markdown("**Performance Limitations:**")
    st.write("• Large texts (>100,000 words) are analyzed using first 5,000-10,000 tokens only")
    st.write("• Analysis results are representative samples, not exhaustive full-text analysis")
    
    st.markdown("**Benchmark & Calibration:**")
    st.write("• **Reading Level:** Calibrated for **U.S. native English speakers**")
    st.write("  - Grade Level 12 = high school senior level")
    st.write("  - Not adjusted for ESL learners or other language backgrounds")
    st.write("• **Sentiment Scores:** No universal benchmark; context-dependent")
    st.write("  - A score of +0.5 may be 'neutral' in reviews but 'positive' in news")

    st.markdown("### 💡 Best Practices & Recommendations")
    st.write("✓ **Use results as exploratory analysis**, not definitive conclusions")
    st.write("✓ **Validate findings** with manual reading and subject matter experts")
    st.write("✓ **Adjust thresholds** based on your specific domain and use case")
    st.write("✓ **Consider context**: cultural background, time period, intended audience")
    st.write("✓ **For critical applications**: Use multiple sentiment analysis tools and human validation")
    st.write("✓ **Non-English text**: Seek language-specific NLP models (e.g., SpaCy, Transformers)")
    
    st.markdown("### 📚 References")
    st.write("• Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.")
    st.write("• Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology.")
    st.write("• NLTK Project: https://www.nltk.org/")
    st.write("• Textstat Library: https://pypi.org/project/textstat/")

# Initialize and run
if __name__ == "__main__":
    main()