import os

import googleapiclient.discovery
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

from config import YT_DOWNLOAD_ABSOLUTE_PATH

load_dotenv()

with open(os.path.join(YT_DOWNLOAD_ABSOLUTE_PATH, 'yt_playlists.txt'), 'r') as playlist_file:
    for playlist_url in playlist_file.read().split("\n"):
        query = parse_qs(urlparse(playlist_url).query, keep_blank_values=True)
        playlist_id = query["list"][0]
        print(f'get all playlist video links from {playlist_id}')
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv('DEVELOPER_KEY'))
        request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=50)
        response = request.execute()

        playlist_items = []

        while request is not None:
            response = request.execute()
            playlist_items += response["items"]
            request = youtube.playlistItems().list_next(request, response)

        print(f"total: {len(playlist_items)}")
        with open(os.path.join(YT_DOWNLOAD_ABSOLUTE_PATH, 'yt_urls.txt'), 'a+') as url_file:
            for item in playlist_items:
                url_file.write(f'https://www.youtube.com/watch?v={item["snippet"]["resourceId"]["videoId"]}\n')
