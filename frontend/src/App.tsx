import React from 'react';
import { SessionProvider } from './context/SessionContext';
import { SessionSidebar } from './components/sidebar/SessionSidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import './styles/global.css';

export const App: React.FC = () => {
  return (
    <SessionProvider>
      <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden', background: '#111214' }}>
        <SessionSidebar />
        <main style={{ flex: 1, height: '100%', overflow: 'hidden' }}>
          <ChatContainer />
        </main>
      </div>
    </SessionProvider>
  );
};

export default App;
