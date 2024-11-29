import sys
import os
import json
import yt_dlp as youtube_dl
import pandas as pd
import plotly.express as px
from collections import Counter
import re
import spacy
from datetime import datetime, timedelta

# Load spaCy's English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Missing model 'en_core_web_sm'. Please run 'python -m spacy download en_core_web_sm' to download it.")
    sys.exit(1)

# Define stopwords set (personal pronouns and prepositions)
STOPWORDS = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'in', 'at', 'on', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just'
}

def get_video_info(video_url, max_age_years=5):
    """
    Use yt-dlp to get YouTube video information and check if the video is within the specified maximum age.
    """
    ydl_opts = {
        'format': 'best',  # Select the best format
        'simulate': True,  # Simulate download process, do not actually download the video
        'writeinfojson': False,  # Do not write info.json file
        'quiet': True,  # Reduce yt-dlp output
        'no_warnings': True,  # Suppress warnings
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except youtube_dl.utils.DownloadError as e:
        print(f"Download error: {e} - Skipping {video_url}")
        return None
    except Exception as e:
        print(f"Unknown error: {e} - Skipping {video_url}")
        return None

    if info is None:
        print(f"Cannot extract video information: {video_url}")
        return None

    # Extract upload date
    upload_date_str = info.get('upload_date')
    if not upload_date_str:
        print(f"Missing upload date information: {video_url} - Skipping")
        return None

    # Convert 'upload_date' to datetime object
    try:
        upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
    except ValueError:
        print(f"Upload date format error: {upload_date_str} - Skipping {video_url}")
        return None

    # Calculate video age
    current_date = datetime.now()
    max_age = timedelta(days=365 * max_age_years)
    video_age = current_date - upload_date

    if video_age > max_age:
        print(f"Video upload date exceeds {max_age_years} years: {upload_date.date()} - Skipping {video_url}")
        return None

    # Extract video information, including channel name
    video_info = {
        'channel': info.get('uploader', 'Unknown'),  # Channel name
        'title': info.get('title', 'Unknown'),  # Video title
        'url': video_url,  # Video URL
        'tags': info.get('tags', []),  # List of tags, default to empty list
        'view_count': info.get('view_count', 0),  # View count, default to 0
        'like_count': info.get('like_count', 0),  # Like count, default to 0
        'upload_date': upload_date.strftime('%Y-%m-%d')  # Formatted upload date
    }
    return video_info

def save_video_info(video_info_list, json_file, excel_file):
    """
    Save video information to JSON and Excel files.
    """
    # Save to JSON file
    existing_data = []
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: {json_file} file is empty or incorrectly formatted, reinitializing data.")
                existing_data = []
    else:
        existing_data = []

    # Add only non-duplicate video information
    existing_urls = {video['url'] for video in existing_data}
    new_videos = [video for video in video_info_list if video['url'] not in existing_urls]
    if new_videos:
        existing_data.extend(new_videos)
    else:
        print("No new video information to save.")

    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    # Save to Excel file
    try:
        df = pd.DataFrame(existing_data)
        # Convert tag list to comma-separated string
        df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
        # Reorder columns
        df = df[['channel', 'title', 'url', 'tags', 'view_count', 'like_count', 'upload_date']]
        # Save to Excel
        df.to_excel(excel_file, index=False)
        print(f"Video information has been saved to Excel file: {excel_file}")
    except Exception as e:
        print(f"Error occurred while saving Excel file: {e}")

def tokenize(text):
    """
    Tokenization function that converts text into a list of words.
    Removes punctuation, converts to lowercase, excludes personal pronouns and prepositions, and retains nouns and adjectives.
    """
    # Use regular expressions to remove punctuation and convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    doc = nlp(text)
    words = []
    for token in doc:
        if token.text in STOPWORDS:
            continue
        if token.pos_ in {'NOUN', 'PROPN', 'ADJ'}:
            words.append(token.lemma_)  # Lemmatize
    return words

def generate_tag_scatter_plot(json_file, output_html='scatter_plot.html'):
    """
    Generate a scatter plot of tag occurrences vs. total view counts, and save as HTML format.
    """
    # Read data from JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    tag_view_counts = {}
    tag_counts = Counter()

    for video in data:
        tags = video.get('tags', [])
        view_count = video.get('view_count', 0)
        for tag in tags:
            tag_counts[tag] += 1
            tag_view_counts[tag] = tag_view_counts.get(tag, 0) + view_count

    if not tag_counts:
        print("No tag data available for plotting.")
        return

    # Prepare data
    tags = list(tag_counts.keys())
    counts = [tag_counts[tag] for tag in tags]
    view_counts = [tag_view_counts[tag] for tag in tags]

    # Create DataFrame
    tag_df = pd.DataFrame({
        'Tag': tags,
        'Count': counts,
        'Total View Count': view_counts
    })

    # Sort by occurrences
    tag_df = tag_df.sort_values(by='Count', ascending=False)

    # Create scatter plot
    fig = px.scatter(tag_df, x='Count', y='Total View Count', hover_data=['Tag'],
                     title='Tag Occurrences vs. Total View Count',
                     labels={'Count': 'Tag Occurrences', 'Total View Count': 'Total View Count'},
                     size='Count', size_max=60)

    # Enhance interactivity
    fig.update_traces(marker=dict(color='blue', opacity=0.6),
                      selector=dict(mode='markers'))

    # Increase figure size
    fig.update_layout(width=1000, height=600, title_font_size=24,
                      xaxis_title_font_size=18,
                      yaxis_title_font_size=18)

    # Save scatter plot as HTML
    fig.write_html(output_html)
    print(f"Scatter plot saved as HTML file: {output_html}")

    # Show figure
    fig.show()

def generate_tag_bar_chart(json_file, output_html='bar_chart.html'):
    """
    Generate a bar chart of tag occurrences, and save as HTML format.
    """
    # Read data from JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    tag_counts = Counter()

    for video in data:
        tags = video.get('tags', [])
        for tag in tags:
            tag_counts[tag] += 1

    if not tag_counts:
        print("No tag data available for plotting.")
        return

    # Prepare data
    tags, counts = zip(*tag_counts.most_common(20))  # Select top 20 tags by occurrences

    # Create DataFrame
    bar_df = pd.DataFrame({
        'Tag': tags,
        'Occurrences': counts
    })

    # Create bar chart
    fig = px.bar(bar_df, x='Tag', y='Occurrences', hover_data=['Occurrences'],
                 title='Top 20 Tag Occurrences',
                 labels={'Occurrences': 'Occurrences', 'Tag': 'Tags'},
                 text='Occurrences')

    # Increase figure size
    fig.update_layout(width=1200, height=700, title_font_size=24,
                      xaxis_title_font_size=18,
                      yaxis_title_font_size=18)

    # Enhance interactivity
    fig.update_traces(marker_color='skyblue',
                      texttemplate='%{text}',
                      textposition='outside')

    # Save bar chart as HTML
    fig.write_html(output_html)
    print(f"Bar chart saved as HTML file: {output_html}")

    # Show figure
    fig.show()

def generate_title_word_stats(json_file, output_html_scatter='title_words_scatter.html', output_html_bar='title_words_bar.html'):
    """
    Generate statistical charts (scatter plot and bar chart) for words in video titles, and save as HTML format.
    """
    # Read data from JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    word_view_counts = {}
    word_counts = Counter()

    for video in data:
        title = video.get('title', '')
        view_count = video.get('view_count', 0)
        words = tokenize(title)
        for word in words:
            word_counts[word] += 1
            word_view_counts[word] = word_view_counts.get(word, 0) + view_count

    if not word_counts:
        print("No title word data available for plotting.")
        return

    # Prepare data
    words = list(word_counts.keys())
    counts = [word_counts[word] for word in words]
    view_counts = [word_view_counts[word] for word in words]

    # Create DataFrame
    word_df = pd.DataFrame({
        'Word': words,
        'Count': counts,
        'Total View Count': view_counts
    })

    # Sort by occurrences
    word_df = word_df.sort_values(by='Count', ascending=False)

    # Create scatter plot
    fig_scatter = px.scatter(word_df, x='Count', y='Total View Count', hover_data=['Word'],
                             title='Word Occurrences in Titles vs. Total View Count',
                             labels={'Count': 'Word Occurrences', 'Total View Count': 'Total View Count'},
                             size='Count', size_max=60)

    # Enhance interactivity
    fig_scatter.update_traces(marker=dict(color='green', opacity=0.6),
                              selector=dict(mode='markers'))

    # Increase figure size
    fig_scatter.update_layout(width=1000, height=600, title_font_size=24,
                              xaxis_title_font_size=18,
                              yaxis_title_font_size=18)

    # Save scatter plot as HTML
    fig_scatter.write_html(output_html_scatter)
    print(f"Title words scatter plot saved as HTML file: {output_html_scatter}")

    # Show figure
    fig_scatter.show()

    # Prepare data for bar chart
    top_words, top_counts = zip(*word_counts.most_common(20))  # Select top 20 words by occurrences

    # Create DataFrame
    bar_word_df = pd.DataFrame({
        'Word': top_words,
        'Occurrences': top_counts
    })

    # Create bar chart
    fig_bar = px.bar(bar_word_df, x='Word', y='Occurrences', hover_data=['Occurrences'],
                     title='Top 20 Word Occurrences in Titles',
                     labels={'Occurrences': 'Occurrences', 'Word': 'Words'},
                     text='Occurrences')

    # Increase figure size
    fig_bar.update_layout(width=1200, height=700, title_font_size=24,
                          xaxis_title_font_size=18,
                          yaxis_title_font_size=18)

    # Enhance interactivity
    fig_bar.update_traces(marker_color='lightgreen',
                          texttemplate='%{text}',
                          textposition='outside')

    # Save bar chart as HTML
    fig_bar.write_html(output_html_bar)
    print(f"Title words bar chart saved as HTML file: {output_html_bar}")

    # Show figure
    fig_bar.show()

def process_youtube_links(txt_file, json_file, excel_file):
    """
    Process YouTube links, extract video information, and generate charts.
    """
    video_info_list = []

    # Check if txt file exists
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} file does not exist.")
        return

    # Read YouTube links from txt file
    with open(txt_file, 'r', encoding='utf-8') as file:
        urls = file.readlines()

    # Strip newline characters from each line
    urls = [url.strip() for url in urls if url.strip()]

    if not urls:
        print(f"Warning: No valid YouTube links in {txt_file} file.")
        return

    # Get video information for each link
    for url in urls:
        video_info = get_video_info(url)
        if video_info:
            video_info_list.append(video_info)

    if not video_info_list:
        print("No valid video information to save.")
        return

    # Save all video information to JSON and Excel files
    save_video_info(video_info_list, json_file, excel_file)

    # Generate charts about tags and save as HTML format
    generate_tag_scatter_plot(json_file)
    generate_tag_bar_chart(json_file)

    # Generate statistical charts for title words and save as HTML format
    generate_title_word_stats(json_file)

# Example usage
if __name__ == "__main__":
    txt_file = 'youtube_links.txt'  # File path of the txt file containing YouTube links
    json_file = 'info.json'  # File path to save video information in JSON
    excel_file = 'video_info.xlsx'  # File path to save video information in Excel
    process_youtube_links(txt_file, json_file, excel_file)

