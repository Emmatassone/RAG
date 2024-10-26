import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { MessageCircle, X } from 'lucide-react';

const Message = PropTypes.shape({
  id: PropTypes.number.isRequired,
  text: PropTypes.string.isRequired,
  sender: PropTypes.oneOf(['user', 'bot']).isRequired,
});

function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user'
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputMessage('');

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from the server');
      }

      const data = await response.json();
      const botMessage = {
        id: Date.now(),
        text: data.response,
        sender: 'bot'
      };

      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const chatBotStyles = {
    container: {
      position: 'fixed',
      bottom: '1rem',
      right: '1rem',
      zIndex: 50,
    },
    card: {
      width: '20rem',
      height: '24rem',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'white',
      borderRadius: '0.5rem',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    },
    header: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1rem',
      borderBottom: '1px solid #e5e7eb',
    },
    title: {
      fontSize: '1.125rem',
      fontWeight: 600,
    },
    closeButton: {
      background: 'none',
      border: 'none',
      cursor: 'pointer',
    },
    messageArea: {
      flexGrow: 1,
      padding: '1rem',
      overflowY: 'auto',
    },
    message: {
      marginBottom: '0.5rem',
      padding: '0.5rem',
      borderRadius: '0.375rem',
      maxWidth: '80%',
    },
    userMessage: {
      backgroundColor: '#3b82f6',
      color: 'white',
      marginLeft: 'auto',
    },
    botMessage: {
      backgroundColor: '#e5e7eb',
      marginRight: 'auto',
    },
    form: {
      padding: '1rem',
      borderTop: '1px solid #e5e7eb',
    },
    input: {
      width: '100%',
      padding: '0.5rem',
      borderRadius: '0.375rem',
      border: '1px solid #d1d5db',
      marginBottom: '0.5rem',
    },
    button: {
      width: '100%',
      padding: '0.5rem',
      backgroundColor: '#3b82f6',
      color: 'white',
      border: 'none',
      borderRadius: '0.375rem',
      cursor: 'pointer',
    },
    openButton: {
      position: 'fixed',
      bottom: '1rem',
      right: '1rem',
      width: '3rem',
      height: '3rem',
      borderRadius: '50%',
      backgroundColor: '#3b82f6',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      border: 'none',
      cursor: 'pointer',
    },
  };

  return (
    <div style={chatBotStyles.container}>
      {isOpen ? (
        <div style={chatBotStyles.card}>
          <div style={chatBotStyles.header}>
            <h2 style={chatBotStyles.title}>Chat Bot</h2>
            <button style={chatBotStyles.closeButton} onClick={() => setIsOpen(false)}>
              <X size={16} />
            </button>
          </div>
          <div style={chatBotStyles.messageArea}>
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  ...chatBotStyles.message,
                  ...(message.sender === 'user' ? chatBotStyles.userMessage : chatBotStyles.botMessage),
                }}
              >
                {message.text}
              </div>
            ))}
          </div>
          <form onSubmit={sendMessage} style={chatBotStyles.form}>
            <input
              type="text"
              placeholder="Type a message..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              style={chatBotStyles.input}
            />
            <button type="submit" style={chatBotStyles.button}>Send</button>
          </form>
        </div>
      ) : (
        <button
          style={chatBotStyles.openButton}
          onClick={() => setIsOpen(true)}
        >
          <MessageCircle size={24} />
        </button>
      )}
    </div>
  );
}

ChatBot.propTypes = {
  messages: PropTypes.arrayOf(Message),
};

export default ChatBot;