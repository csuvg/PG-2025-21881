"""
Upload files to Google Drive with progress tracking
Usage: python upload_to_drive.py <file_path>
"""

import os
import sys
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_credentials():
    """Get or refresh credentials for Google Drive API"""
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            
            # For headless environments (like RunPod), use manual flow
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print("\n" + "="*70)
            print("🔐 AUTHENTICATION REQUIRED")
            print("="*70)
            print("\n1. Open this URL in your browser (on any device):")
            print(f"\n   {auth_url}\n")
            print("2. Authorize the application")
            print("3. Copy the authorization code")
            print("="*70 + "\n")
            
            code = input("📋 Paste the authorization code here: ").strip()
            
            flow.fetch_token(code=code)
            creds = flow.credentials
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
        
        print("✅ Authentication successful! Token saved for future use.\n")
    
    return creds

def upload_file_with_progress(file_path, folder_id=None):
    """Upload a file to Google Drive with progress tracking"""
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    print(f"\n📁 File: {file_name}")
    print(f"📊 Size: {file_size / (1024*1024):.2f} MB")
    print(f"🚀 Starting upload to Google Drive...\n")
    
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink')
    
    response = None
    last_progress = 0
    
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            if progress != last_progress:
                bar_length = 40
                filled_length = int(bar_length * status.progress())
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f'\r⬆️  [{bar}] {progress}% ', end='', flush=True)
                last_progress = progress
    
    print(f'\n\n✅ Upload complete!')
    print(f'🔗 File ID: {response.get("id")}')
    print(f'🌐 View link: {response.get("webViewLink")}')
    
    return response

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_drive.py <file_path> [folder_id]")
        print("\nExample: python upload_to_drive.py my_file.zip")
        print("Example: python upload_to_drive.py my_file.zip 1a2b3c4d5e6f7g8h9i")
        sys.exit(1)
    
    file_path = sys.argv[1]
    folder_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        sys.exit(1)
    
    try:
        upload_file_with_progress(file_path, folder_id)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

