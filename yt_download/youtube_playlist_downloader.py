import os

import googleapiclient.discovery
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

from config import YT_DOWNLOAD_ABSOLUTE_PATH


def remove_duplicates():
    lines_seen = set()  # holds lines already seen
    with open(os.path.join(YT_DOWNLOAD_ABSOLUTE_PATH, 'yt_urls.txt'), "w") as output_file:
        for each_line in open(os.path.join(YT_DOWNLOAD_ABSOLUTE_PATH, 'yt_urls_duplicates.txt'), "r"):
            if each_line not in lines_seen:  # check if line is not duplicate
                output_file.write(each_line)
                lines_seen.add(each_line)


def urls_from_playlists():
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
            with open(os.path.join(YT_DOWNLOAD_ABSOLUTE_PATH, 'yt_urls_duplicates.txt'), 'a+') as url_file:
                for item in playlist_items:
                    url_file.write(f'https://www.youtube.com/watch?v={item["snippet"]["resourceId"]["videoId"]}\n')


if __name__ == '__main__':
    load_dotenv()
    urls_from_playlists()
    remove_duplicates()
