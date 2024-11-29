# YouTube Video Analysis & Visualization Tool

This tool analyzes YouTube videos to uncover patterns in viewership and engagement by examining video tags and titles using natural language processing techniques.

## Overview

Ever wondered what makes certain YouTube videos go viral while others struggle for views? This YouTube Video Analysis Tool helps discover what topics attract the most views by analyzing video tags and titles. The tool uses natural language processing to dissect video titles and tags to look into the patterns that influence viewership.

### Key Features

- Automated data collection from multiple YouTube channels
- NLP-based text analysis of titles and tags
- Interactive visualizations of engagement patterns
- Trend analysis across different time periods
- Export data to Excel for further analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/youtube-analysis-tool.git
cd youtube-analysis-tool
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Create a text file named `youtube_links.txt` containing YouTube video URLs (one per line)

2. Run the analysis:
```bash
python youtube_analysis.py
```

3. The script will generate several output files:
- `info.json`: Raw data in JSON format
- `video_info.xlsx`: Processed data in Excel format
- `tag_scatter.html`: Interactive visualization of tag occurrences vs. views
- `tag_bar.html`: Top 20 most common tags
- `title_words_scatter.html`: Word occurrences in titles vs. views
- `title_words_bar.html`: Top 20 most common words in titles

## Example Output

The `example` directory contains sample output files demonstrating the tool's capabilities. These include visualizations of tag usage patterns and their correlation with view counts.

## Requirements

- Python 3.7+
- yt-dlp
- pandas
- plotly
- spaCy
- See `requirements.txt` for complete list

## License

This project is licensed under the MIT License.