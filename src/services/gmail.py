"""
Gmail Service Adapter

Wraps Gmail API for fetching attachments and managing messages.
Provides clean data object interfaces without printing to stdout.
"""

import os
import base64
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import BatchHttpRequest
except ImportError:
    raise ImportError(
        "Google API libraries not installed. "
        "Install with: pip install google-auth google-auth-oauthlib google-api-python-client"
    )

from ..config_loader import config

logger = logging.getLogger(__name__)


@dataclass
class GmailAttachment:
    """Data object representing a Gmail attachment."""
    message_id: str
    attachment_id: str
    filename: str
    body_text: Optional[str] = None


@dataclass
class AttachmentData:
    """Data object for downloaded attachment with metadata."""
    message_id: str
    attachment_id: str
    filename: str
    raw_bytes: bytes
    body_text: Optional[str] = None


class GmailConnector:
    """
    Gmail API connector for message and attachment operations.
    
    Handles authentication, querying, batch fetching, and cleanup.
    All methods return data objects rather than printing to stdout.
    """
    
    def __init__(
        self,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        scopes: Optional[List[str]] = None
    ):
        """
        Initialize Gmail connector.
        
        Args:
            credentials_file: Path to OAuth credentials JSON
            token_file: Path to store/load access token
            scopes: Gmail API scopes (defaults to config value)
        """
        self.credentials_file = credentials_file or config.get('gmail.credentials.file', 'credentials.json')
        self.token_file = token_file or config.get('gmail.credentials.token_file', 'token.json')
        self.scopes = scopes or config.get('gmail.scopes', ["https://www.googleapis.com/auth/gmail.modify"])
        self.batch_uri = config.get('gmail.batch_uri', "https://gmail.googleapis.com/batch/gmail/v1")
        
        self._service = None
    
    def get_service(self):
        """
        Get authenticated Gmail API service.
        
        Handles OAuth flow and token refresh automatically.
        
        Returns:
            Gmail API service resource
            
        Raises:
            FileNotFoundError: If credentials file not found
            HttpError: If authentication fails
        """
        if self._service:
            return self._service
        
        logger.info("Initializing Gmail API service")
        
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
                logger.debug(f"Loaded credentials from {self.token_file}")
            except Exception as e:
                logger.warning(f"Failed to load token file: {e}")
        
        # Refresh or authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                logger.info("Starting OAuth flow")
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_file}\n"
                        "Download from Google Cloud Console"
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, 
                    self.scopes
                )
                oauth_port = config.get('gmail.credentials.oauth_port', 8080)
                creds = flow.run_local_server(port=oauth_port)
            
            # Save credentials
            try:
                with open(self.token_file, "w") as token:
                    token.write(creds.to_json())
                logger.info(f"Saved credentials to {self.token_file}")
            except Exception as e:
                logger.warning(f"Failed to save token: {e}")
        
        # Build service
        try:
            self._service = build("gmail", "v1", credentials=creds)
            logger.info("Gmail API service initialized successfully")
            return self._service
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {e}")
            raise
    
    def fetch_attachments_with_pattern(
        self,
        query: str,
        filename_pattern: str,
        extract_body_text: bool = False
    ) -> List[GmailAttachment]:
        """
        Fetch attachments matching query and filename pattern.
        
        Args:
            query: Gmail search query
            filename_pattern: Regex pattern for attachment filenames
            extract_body_text: Whether to extract plain text from message body
            
        Returns:
            List of GmailAttachment objects
        """
        import re
        
        service = self.get_service()
        
        logger.info(f"Searching for messages: query='{query}', pattern='{filename_pattern}'")
        
        try:
            results = service.users().messages().list(userId="me", q=query).execute()
            messages = results.get("messages", [])
            
            if not messages:
                logger.info(f"No messages found for query: {query}")
                return []
            
            logger.debug(f"Found {len(messages)} message(s) matching query")
            
        except HttpError as e:
            logger.error(f"Failed to list messages: {e}")
            return []
        
        # Batch fetch message details
        matches = []
        
        def callback(request_id, response, exception):
            if exception:
                logger.error(f"Error fetching message {request_id}: {exception}")
                return
            
            # Extract body text if requested
            body_text = None
            if extract_body_text:
                body_text = self._extract_plain_text_from_msg(response)
            
            # Find matching attachments
            parts = response.get("payload", {})
            attachments = self._extract_attachments_recursive(parts)
            
            for att in attachments:
                if re.search(filename_pattern, att["filename"], re.IGNORECASE):
                    matches.append(GmailAttachment(
                        message_id=response["id"],
                        attachment_id=att["attachmentID"],
                        filename=att["filename"],
                        body_text=body_text
                    ))
        
        # Execute batch request
        batch = BatchHttpRequest(callback=callback, batch_uri=self.batch_uri)
        for msg in messages:
            batch.add(service.users().messages().get(userId="me", id=msg["id"]))
        
        try:
            batch.execute()
            logger.info(f"Found {len(matches)} matching attachment(s)")
        except Exception as e:
            logger.error(f"Batch request failed: {e}")
        
        return matches
    
    def download_attachment(self, attachment: GmailAttachment) -> Optional[AttachmentData]:
        """
        Download attachment data from Gmail.
        
        Args:
            attachment: GmailAttachment object with message and attachment IDs
            
        Returns:
            AttachmentData with raw bytes, or None if download fails
        """
        service = self.get_service()
        
        try:
            result = service.users().messages().attachments().get(
                userId="me",
                messageId=attachment.message_id,
                id=attachment.attachment_id
            ).execute()
            
            raw_bytes = base64.urlsafe_b64decode(result["data"])
            
            logger.debug(
                f"Downloaded attachment: {attachment.filename} "
                f"({len(raw_bytes)} bytes)"
            )
            
            return AttachmentData(
                message_id=attachment.message_id,
                attachment_id=attachment.attachment_id,
                filename=attachment.filename,
                raw_bytes=raw_bytes,
                body_text=attachment.body_text
            )
            
        except HttpError as e:
            logger.error(
                f"Failed to download attachment {attachment.filename}: {e}"
            )
            return None
    
    def trash_messages(self, message_ids: Set[str]) -> Dict[str, bool]:
        """
        Move messages to trash (batch operation).
        
        Args:
            message_ids: Set of Gmail message IDs to trash
            
        Returns:
            Dict mapping message_id -> success boolean
        """
        if not message_ids:
            logger.info("No messages to trash")
            return {}
        
        service = self.get_service()
        results = {}
        
        logger.info(f"Trashing {len(message_ids)} message(s)")
        
        def callback(request_id, response, exception):
            msg_id = request_id
            if exception:
                logger.error(f"Failed to trash message {msg_id}: {exception}")
                results[msg_id] = False
            else:
                logger.debug(f"Trashed message {msg_id}")
                results[msg_id] = True
        
        batch = BatchHttpRequest(callback=callback, batch_uri=self.batch_uri)
        
        for msg_id in message_ids:
            batch.add(
                service.users().messages().trash(userId="me", id=msg_id),
                request_id=msg_id
            )
        
        try:
            batch.execute()
            success_count = sum(1 for v in results.values() if v)
            logger.info(
                f"Trash operation complete: {success_count}/{len(message_ids)} successful"
            )
        except Exception as e:
            logger.error(f"Batch trash operation failed: {e}")
        
        return results
    
    def _extract_plain_text_from_msg(self, msg: Dict) -> str:
        """
        Extract first text/plain part from Gmail message.
        
        Args:
            msg: Gmail message resource
            
        Returns:
            Plain text content or empty string
        """
        def walk(part):
            mime = part.get("mimeType", "")
            
            if mime == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    try:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    except Exception as e:
                        logger.warning(f"Failed to decode message body: {e}")
                        return ""
            
            # Recurse into parts
            for p in part.get("parts", []):
                result = walk(p)
                if result:
                    return result
            
            return ""
        
        payload = msg.get("payload", {})
        return walk(payload)
    
    def _extract_attachments_recursive(self, part: Dict) -> List[Dict[str, str]]:
        """
        Recursively extract attachment metadata from message parts.
        
        Args:
            part: Message part dict
            
        Returns:
            List of dicts with filename and attachmentID
        """
        attachments = []
        
        if part.get("filename") and "attachmentId" in part.get("body", {}):
            attachments.append({
                "filename": part.get("filename"),
                "attachmentID": part["body"]["attachmentId"]
            })
        
        for subpart in part.get("parts", []):
            attachments.extend(self._extract_attachments_recursive(subpart))
        
        return attachments
