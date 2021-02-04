import { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import Head from 'next/head'
import Pusher from 'pusher-js';
import styles from '../styles/Home.module.css'
import { FilePicker, Button, Pane, Text, Heading, Paragraph, Alert, Checkbox, Dialog, IconButton, CogIcon, TextInput, extractStyles  } from 'evergreen-ui';
import Fade from 'react-reveal/Fade';
import ProgressBar from '@ramonak/react-progress-bar';

export default function Home() {

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(null); // set when script runs and updates are received
    const [results, setResults] = useState(null); // set when results received
    const [processedVideoPath, setProcessedVideoPath] = useState(''); // set when script finished
    const [pusher, setPusher] = useState(null);
    const [configDialogShown, setConfigDialogShown] = useState(false);
    const [args, setArgs] = useState([
      { name: '-c', label: 'Confidence', value: '0.4' },
      { name: '-s', label: 'Skip Frames', value: '30' },
      { name: '-d', label: 'Dimension', value: '500' },
      { name: '-a', label: 'Count Direction', value: 'vertical' },
      { name: '-l', label: 'Counting Line Position', value: '0.5' },
      { name: '-m', label: 'Minimum #Frames before Count', value: '0' },
      { name: '-b', label: 'Draw Bounding Boxes', value: 'false' },
      { name: '-f', label: 'Max Disappeared', value: '50' },
      { name: '-j', label: 'Max Distance', value: '50' },
    ]);

    useEffect(() => {
        setPusher(
            new Pusher(
                process.env.NEXT_PUBLIC_PUSHER_APP_KEY, {
                    cluster: process.env.NEXT_PUBLIC_PUSHER_APP_CLUSTER,
                    useTLS: true,
                }
            )
        );
    }, []);

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
        formData.append(
            'args',
            JSON.stringify(args)
        );
        axios.post('/api/analyze', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            timeout: 0,
        })
            .then(res => {
                console.log(res);
                if (pusher) setProgressChannel(pusher.subscribe(res.data.clientId));
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
        <>
            <Pane className={styles.container}>
                <Head>
                    <title>People Counter</title>
                </Head>

                <Pane className={styles.configContainer}>
                    <Heading is='h1' size={900} marginBottom={48}><strong>People Counter</strong></Heading>

                    <Paragraph marginBottom={24}>Upload a .mp4 file to be analyzed.</Paragraph>

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
                            className={styles.submitBtn}
                        >
                            Submit
                        </Button>
                        <Pane className={styles.ownRow}>
                            <IconButton 
                                icon={CogIcon}
                                appearance='minimal'
                                onClick={() => setConfigDialogShown(true)}
                            />
                            <Checkbox 
                                label='Create annotated video.'
                                checked={withVideo}
                                onChange={e => setWithVideo(e.target.checked)}
                                className={styles.check}
                            />
                        </Pane>
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
                            <Pane className={styles.barWrapper}>
                                <ProgressBar 
                                    completed={progress} 
                                    bgcolor='#116AB8'
                                    width='100%' 
                                />
                            </Pane>
                        </Fade>
                    )}
                    
                    { !error && processedVideoPath && (
                        <Fade bottom>
                            <Text className={styles.scrollText}>Scroll to see results.</Text>
                            <video src={processedVideoPath} controls />
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

            <Dialog
                isShown={configDialogShown}
                title='Tweak Arguments'
                onCloseComplete={() => setConfigDialogShown(false)}
                confirmLabel='OK'
                hasCancel={false}
            >   
                { args.map(aconf => (
                    <Pane
                        key={aconf.name}
                        display='flex'
                        alignItems='center'
                        marginY='.5rem'
                    >
                        <Pane flex='1'>{aconf.label}</Pane>
                        <Pane flex='1'>
                            <TextInput
                                width='unset'
                                value={aconf.value}
                                onChange={e => setArgs(prev => prev.map(c => (
                                    c.name === aconf.name 
                                        ? { ...c, value: e.target.value }
                                        : c
                                )))}
                            />
                        </Pane>
                    </Pane>
                ))}
            </Dialog>
        </>
    )
}