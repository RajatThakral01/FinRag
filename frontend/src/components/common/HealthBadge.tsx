import React, { useState, useEffect } from 'react';
import { api } from '../../api/client';
import styles from './HealthBadge.module.css';

export const HealthBadge: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        await api.checkHealth();
        if (isMounted) setIsOnline(true);
      } catch {
        if (isMounted) setIsOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={styles.badge} title={isOnline ? 'Backend API Connected' : 'Backend API Disconnected'}>
      <span
        className={`${styles.dot} ${
          isOnline === true ? styles.online : isOnline === false ? styles.offline : styles.pending
        }`}
      />
      <span className={styles.text}>
        {isOnline === true ? 'API Connected' : isOnline === false ? 'Offline' : 'Connecting...'}
      </span>
    </div>
  );
};
