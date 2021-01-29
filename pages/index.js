import { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import Head from 'next/head'
import Pusher from 'pusher-js';
import styles from '../styles/Home.module.css'
import { FilePicker, Button, Pane, Text, Heading, Paragraph, Alert, Checkbox  } from 'evergreen-ui';
import Fade from 'react-reveal/Fade';
import ProgressBar from '@ramonak/react-progress-bar';

export default function Home() {

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(null); // set when script runs and updates are received
    const [results, setResults] = useState(null); // set when results received
    const [processedVideoPath, setProcessedVideoPath] = useState(''); // set when script finished

    const pusher = useMemo(() => new Pusher(
        process.env.NEXT_PUBLIC_PUSHER_APP_KEY, {
            cluster: process.env.NEXT_PUBLIC_PUSHER_APP_CLUSTER,
            useTLS: true,
        }
    ), []);
    const [progressChannel, setProgressChannel] = useState(null);
    useEffect(() => {
        if (progressChannel) {
            progressChannel.bind('progress', data => {
                setProgress(data.value);
                console.log(data);
            });
            progressChannel.bind('results', data => {
                setProgress(100);
                setResults(data);
                console.log(data)
            })
            progressChannel.bind('success', data => {
                setError(null);
                setProcessedVideoPath(data.videopath);
                console.log(data)
            })
            progressChannel.bind('error', data => {
                setError(data);
                console.log(data)
            })
        }
    }, [progressChannel]);


    const [withVideo, setWithVideo] = useState(true);
    const [file, setFile] = useState(null);
    const onSubmit = () => {
        setLoading(true);
        setError(null);
        setResults(null);
        setProcessedVideoPath('');
        setProgress(null);
        const formData = new FormData();
        formData.append(
            'video',
            file,
        );
        formData.append(
            'withVideo',
            withVideo ? 'true' : 'false'
        );
        axios.post('/api/analyze', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            timeout: 0,
        })
            .then(res => {
                console.log(res);
                setProgressChannel(pusher.subscribe(res.data.clientId));
                setProgress(0);
                setLoading(false);
            })
            .catch(err => {
                console.log(err);
                setError(err);
                setLoading(false);
            });
    }


    return (
        <Pane className={styles.container}>
            <Head>
                <title>People Counter</title>
            </Head>

            <Pane className={styles.configContainer}>
                <Heading is='h1' size={900} marginBottom={48}><strong>People Counter</strong></Heading>

                <Paragraph marginBottom={24}>Upload an .mp4 file to be analyzed.</Paragraph>

                <Pane  className={styles.form}>
                    {/* <input type='file' name='video' onChange={(e) => setFile(e.target.files[0])} /> */}
                    <FilePicker 
                        multiple={false} 
                        accept='.mp4' 
                        name='video' 
                        onChange={(e) => setFile(e[0])}
                    />
                    <Button 
                        marginLeft='3rem' 
                        appearance='primary' 
                        height={40} 
                        isLoading={loading} 
                        disabled={!file || (!error && progress !== null && !results)} 
                        onClick={onSubmit}
                    >
                        Submit
                    </Button>
                    <Checkbox 
                        label='Create annotated video.'
                        checked={withVideo}
                        onChange={e => setWithVideo(e.target.checked)}
                        className={styles.check}
                    />
                </Pane>
            </Pane>

            <Pane className={styles.resultsContainer}>

                { error && (
                    <Alert
                        intent='warning'
                        title='Something went wrong.'
                    />
                )}

                { !error && progress !== null && (
                    <Fade bottom>
                        <ProgressBar 
                            completed={progress} 
                            bgcolor='#116AB8'
                            width='30rem' 
                        />
                    </Fade>
                )}
                
                { !error && processedVideoPath && (
                    <Fade bottom>
                        <Text className={styles.scrollText}>Scroll to see results.</Text>
                        <video src={processedVideoPath} width={700} controls />
                    </Fade>
                )}

                { !error && results && (
                    <Fade bottom>
                        <Pane
                            className={styles.resultsTable}
                        >
                            { results.map(result => (
                                <Pane className={styles.row}>
                                    <Text className={styles.item}>{result.label}</Text>
                                    <Text className={styles.item}>{result.value}</Text>
                                </Pane>
                            ))}
                        </Pane>
                    </Fade>
                )}
                
            </Pane>
            
        </Pane>
    )
}
