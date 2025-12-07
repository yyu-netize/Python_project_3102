import re

class PIIRedactor:
    def __init__(self):
        # Pre-compile regular expressions for efficiency
        
        # 1. Email matching (standard format)
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        
        # 2. IP address matching (standard IPv4)
        self.ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        
        # 3. SSN (Social Security Number) matching (format 000-00-0000)
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        
        # 4. Phone number matching (key point: intentionally designed to be slightly broad)
        # Matches formats like: 123-456-7890, (123) 456-7890, 123.456.7890
        # Note: This may incorrectly match game damage ranges like "100-200-300"
        self.phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

    def redact(self, text):
        """
        Input text, return redacted text
        """
        if not text:
            return ""
            
        # Replace in order with special markers for subsequent statistics
        text = self.email_pattern.sub('<EMAIL_REDACTED>', text)
        text = self.ip_pattern.sub('<IP_REDACTED>', text)
        text = self.ssn_pattern.sub('<SSN_REDACTED>', text)
        text = self.phone_pattern.sub('<PHONE_REDACTED>', text)
        
        return text

# Simple test code
if __name__ == "__main__":
    cleaner = PIIRedactor()
    sample = "Call me at 555-019-2345 or email dave@popcap.com. My IP is 192.168.1.1. Damage range is 100-200-3000."
    print("Original:", sample)
    print("Cleaned :", cleaner.redact(sample))